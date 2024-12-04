import csv
import os
import cv2
import yaml
import utils
import logging
import argparse
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from factory import model_factory
from kitti import KITTITest
import pandas as pd
from models.utils import mesh_grid, knn_interpolation
import glob
import time
import random
import math
import utm
import subprocess
import sys
import shutil
from sklearn.mixture import GaussianMixture
import geopy.distance

# ***************************** START: HELPER FUNCTIONS/WRAPPERS ***************************** 

def get_rigid_transformation(calib_path):
    ''' Returns a rigid transformation matrix in homogeneous coordinates (combination of
        rotation and translation.
        Used to obtain:
            - LiDAR to camera reference transformation matrix 
            - IMU to LiDAR reference transformation matrix
        '''
    with open(calib_path, 'r') as f:
        calib = f.readlines()
    R = np.array([float(x) for x in calib[1].strip().split(' ')[1:]]).reshape((3, 3))
    t = np.array([float(x) for x in calib[2].strip().split(' ')[1:]])[:, None]
    T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
    
    return T

def gaussian_mixture_model(region, n_components=2):
    ''' Returns a Gaussian mixture model probability distribution given the: 
            - Image region
            - Number of mixture components
        '''
    # -- Flatten the image region to 1D array --
    image_data = region.ravel()

    # -- Create a Gaussian Mixture Model --
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(image_data.reshape(-1, 1))  # reshape data to (n_samples, 1)
    means = np.array(gmm.means_)
    std_dev_0 =np.sqrt(gmm.covariances_[0])[0][0]
    std_dev_1 =np.sqrt(gmm.covariances_[1])[0][0]

    return np.array([means[0][0], means[1][0]]), np.array([std_dev_0, std_dev_1]) 

def pool_max(c_input, name, kern, stride, mask_bg, mask_lid):
    ''' Obtain the MAX of the background for the whole matrix (neglects regions 
        of probable background inside bounding boxes) by pooling. Inputs: 
            - Image matrix 
            - Channel
            - Kernel and stride sizes
            - Background and neighbourhood masks
        '''
    # -- Set non background areas to 0 --
    c_input = np.squeeze(c_input, -1)
    c_img_bg = c_input.copy()
    c_img_bg[np.where(mask_bg==0)] = 0

    # -- Convert to tensor for pooling --
    c_img_bg_T    = T.ToTensor()(c_img_bg)

    # -- Average pooling --
    pooling = nn.AvgPool2d(kern, stride = stride)
    c_img_bg_P = pooling(c_img_bg_T)
    c_img_bg_P = c_img_bg_P.squeeze(0)

    # -- Get pooled max values and location & map to original image --
    bg_P_max_loc = np.unravel_index(np.argmax(c_img_bg_P), c_img_bg_P.shape)

    bg_w_s = int(bg_P_max_loc[1]*stride)
    bg_h_s = int(bg_P_max_loc[0]*stride)
    max_abs_bg_img = np.max(c_img_bg[bg_h_s:bg_h_s+kern, bg_w_s:bg_w_s+kern])

    # -- Obtain the max of lidar filtered region from non shifted channels --    
    c_img_lidar = c_input.copy()
    c_img_lidar[np.where(np.squeeze(mask_lid)==0)] = 0

    # -- Convert to tensor for pooling --
    c_img_lid_T    = T.ToTensor()(c_img_lidar)

    # -- Average pooling to get robust maximum value --
    pooling = nn.AvgPool2d(kern, stride = stride)
    c_img_lid_P = pooling(c_img_lid_T)
    c_img_lid_P = c_img_lid_P.squeeze(0)

    # -- Get pooled max values and location & map to original image --
    lid_P_max_loc = np.unravel_index(np.argmax(c_img_lid_P), c_img_lid_P.shape)
    lid_w_s = int(lid_P_max_loc[1]*stride)
    lid_h_s = int(lid_P_max_loc[0]*stride)
    max_abs_fg_img = np.max(c_img_lidar[lid_h_s:lid_h_s+kern, lid_w_s:lid_w_s+kern])

    return max_abs_bg_img, [bg_h_s, bg_w_s], max_abs_fg_img, [lid_h_s, lid_w_s]

def isRectangleOverlap(R2, R1):
    ''' Returns the indexes of the overlapping corners given two regions
        '''
    #Note: R1 is always the current region
    # -- Function to check if (corner) point is within the other region --
    def point_within_region(point, region):
        x, y = point
        x1, y1, x2, y2 = region
        min_y = np.min([y1, y2])
        max_y = np.max([y1, y2])
        min_x = np.min([x1, x2])
        max_x = np.max([x1, x2])
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    # -- Array to store overlapped corner indexes --
    overlapped_corns = []

    # -- No overlap scenario --
    if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
        pass
    # -- Overlap scenario --
    else:
        corner_points_R2 = [(R2[0], R2[1]), (R2[0], R2[3]), (R2[2], R2[1]), (R2[2], R2[3])]

        # -- Loop all corner points and see which of it is in the other region --
        for count, point in enumerate(corner_points_R2):
            if point_within_region(point, R1):
                overlapped_corns.append(count)
                
    return overlapped_corns

def calculate_overlap_ratio(bb_1, bb_2):
    ''' Returns the overlapping ratio of two regions
        '''
    # -- Extract coordinates of bb_1 and bb_2 --
    x1_1, y1_1, x2_1, y2_1 = bb_1
    x1_2, y1_2, x2_2, y2_2 = bb_2

    # -- Calculate the area of bb_1 --
    area_bb_1 = (x2_1 - x1_1) * (y2_1 - y1_1)

    # -- Calculate the intersection area --
    x1_intersection = max(x1_1, x1_2)
    y1_intersection = max(y1_1, y1_2)
    x2_intersection = min(x2_1, x2_2)
    y2_intersection = min(y2_1, y2_2)

    # -- Check if there is no overlap --
    if x1_intersection >= x2_intersection or y1_intersection >= y2_intersection:
        return 0  # No overlap, percentage is 0

    # -- Calculate the area of the intersection --
    intersection_area = (x2_intersection - x1_intersection) * (y2_intersection - y1_intersection)

    # -- Calculate the percentage of bb_1 within bb_2 --
    percentage_within = (intersection_area / area_bb_1)

    return percentage_within

def preprocess_speed(odo_file, direction):
    ''' Returns the speed value of a given direction from the Odometry data
        '''
    with open(odo_file,'r') as f:
        vals=f.read()
        vals=vals.split()
        spd = float(vals[direction])
        return spd

def read_calib_to_csv(calib_file, delimiter):
    ''' Reads a calibration data file and returns its csv equivalent
        '''
    max_col = 0
    with open(calib_file, 'r') as f_temp:
        lines = f_temp.readlines()
    for count in lines:
        col_count = len(count.split(delimiter)) + 1
        if (max_col < col_count):
            max_col = col_count
    cols = [count for count in range(0, max_col)]
    data_csv = pd.read_csv(calib_file, delimiter=delimiter, header=None, index_col=0, names=cols)
    return data_csv

def pointcloud_to_image(ptcloud, imwidth, imheight, tr, P0, max_depth, labels_file):
    """
    Maps the point cloud to the left camera frame
    Args:
        ptcloud: velodyne data 
        imwidth, imheight: image dimensions
        tr: transformation matrix
        P0: left camera calibration
        max_depth: depth w.r.t. ego vehicle within which motion will be analyzed
        labels_file: path to region annotations
    Returns:
        Image of mapped point cloud with dimensions = imheight x imwidth
    """
    # -- Eliminate all points behind the lidar (x val less than 1) --
    ptcloud = ptcloud[ptcloud[:,0] > 0]

    # -- Drop reflectance info (last column) and replace with ones (for homogenity) - X,Y,Z,1 --
    ptcloud = np.hstack([ptcloud[:, :3], np.ones(ptcloud.shape[0]).reshape((-1,1))])

    # -- Transform pointcloud to camera coordinate frame --
    cam_projection = tr.dot(ptcloud.T)

    # -- Redo point elimination behind the camera (for robustness) --
    cam_projection = cam_projection[:, cam_projection[2] > 0]

    # -- Extract the depth from the camera --
    cam_depth = cam_projection[2].copy()

    # -- Project to plane (2D) --
    cam_projection = cam_projection / cam_depth

    # -- Make 3D coordinates homogenous - X,Y,Z,1
    cam_projection = np.vstack([cam_projection, np.ones(cam_projection.shape[1])])

    # -- Get pixel coordinates (round float values) --
    cam_projection = P0.dot(cam_projection)

    pixel_coords = np.round(cam_projection.T, 0)[:, :2].astype('int')

    # -- Limit values to image frame dimensions --
    pixel_indices = np.where((pixel_coords[:, 0] <  imwidth)
                &(pixel_coords[:, 0] >= 0)
                &(pixel_coords[:, 1] <  imheight)
                &(pixel_coords[:, 1] >= 0))
    pixel_coords = pixel_coords[pixel_indices]
    cam_depth = cam_depth[pixel_indices]

    # -- Create 0 matrix for blank image --
    cloud_img = np.zeros( (imheight, imwidth) )

    # -- Fill the empty image --
    for item in enumerate(pixel_coords):
        idx = item[0]
        (u, v) = item[1]
        # -- Check dimensions (for robustness) and allocate values --
        if ( (u >= 0) and (u < imwidth) ):
            if ( (v >= 0) and (v < imheight) ):
                cloud_img[v, u] = cam_depth[idx]

    # -- Fill in 0 values with large dist so that they will be ignored --
    cloud_img[cloud_img == 0.0] = max_depth 

    # -- Load the regions for overlaps checking --
    bboxes = []
    labels = csv.reader(open(labels_file), delimiter=" ")
    for label in labels:
        # -- Select ROI i.e. region within bounding box --
        bboxes.append([int(float(label[1])), int(float(label[2])), int(float(label[3])), int(float(label[4]))])

    return cloud_img

def timestamps2seconds(timestamp_path):
    ''' Reads a timestamp and returns its conversion in seconds
        '''
    # -- Use pandas to read the text file as a csv --
    df = pd.read_csv(timestamp_path, header=None, names=["Timestamp"])
    
    # -- Get Hours, Minutes, and Seconds --
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    hours = df['Timestamp'].dt.hour
    minutes = df['Timestamp'].dt.minute
    seconds = df['Timestamp'].dt.second + df['Timestamp'].dt.microsecond / 1e6
    
    # -- Create a numpy array with hours, minutes, and seconds --
    hms_vals = np.vstack((hours, minutes, seconds)).T
    
    # -- Calculate total seconds from hours, minutes, and seconds --
    def calculate_total_seconds(hms):
        return hms[0] * 3600 + hms[1] * 60 + hms[2]
    total_seconds = np.array(list(map(calculate_total_seconds, hms_vals)))
    
    # -- Return total_seconds as a numpy array --
    return total_seconds

def imu_to_geodetic(x, y, z, lat_ego, lon_ego, alt_ego, heading_ego):
  """
    Convert IMU coordinates to geodetic coordinates.
    Args:
        x, y, z: IMU values of the road user's 3D center relative to the ego vehicle.
        lat_ego, long_ego, alt_ego, heading_ego: latitude (deg), longitude (deg), altitude (m) and heading (rad)
    Returns:
        A tuple (latitude, longitude, altitude) representing the geodetic coordinates.
    """
  easting, northing, utm_zone_number, utm_zone_letter = utm.from_latlon(lat_ego, lon_ego)
  easting += x * math.cos(heading_ego) - y * math.sin(heading_ego)
  northing += x * math.sin(heading_ego) + y * math.cos(heading_ego)
  geodetic_altitude = alt_ego + z
  geodetic_latitude, geodetic_longitude = utm.to_latlon(easting, northing, utm_zone_number, utm_zone_letter)
  geodetic_coords = [geodetic_latitude, geodetic_longitude, geodetic_altitude]
  return geodetic_coords

def cam_to_imu(location, T_ref0_ref2, T_velo_ref0, T_imu_velo):
    """
    Convert camera coordinates to IMU coordinates.
    Args:
        location: a tuple of coordinates w.r.t camera frame
        T_ref0_ref2, T_velo_ref0, T_imu_velo: transformation matrices
    Returns:
        A tuple of coordinates w.r.t IMU frame
    """
    loc_shaped = []
    loc_shaped.append(location) 
    loc_shaped = np.array(loc_shaped)
    loc_shaped_homo = np.hstack((loc_shaped, np.ones((loc_shaped.shape[0], 1)))) 
    c_ref2_ref0 = np.dot(np.linalg.inv(T_ref0_ref2), loc_shaped_homo.T) # transform from cam 2 to cam 0
    c_ref0_velo = np.dot(np.linalg.inv(T_velo_ref0), c_ref2_ref0)       # transform from cam 0 to velo
    imuT_xyz_homo = np.dot(np.linalg.inv(T_imu_velo), c_ref0_velo).T    # transform from velo to imu
    location_imu_xyz = imuT_xyz_homo[:, :-1][0]
    return location_imu_xyz

def check_motion(current_odo, prev_odo, track_labels_file, test_id, dist_thresh, T_ref0_ref2, T_velo_ref0, T_imu_velo, timestep):
    """
    Get the motion status of all tracks
    Args:
        current_odo, prev_odo: odometry file of current and previous frames
        track_labels_file: file of generated track data
        test id, timestep: test frame id and time step value
        dist_thresh: distance threshold for motion estimation
        T_ref0_ref2, T_velo_ref0, T_imu_velo: transformation matrices
    Returns:
        A tuple of coordinates w.r.t IMU frame
    """
    # -- read current odo file --
    with open(current_odo,'r') as f1:
        current_odo_vals=f1.read()
        current_odo_vals=current_odo_vals.split()
        current_lat_ego = float(current_odo_vals[0]) #deg
        current_lon_ego = float(current_odo_vals[1]) #deg
        current_alt_ego = float(current_odo_vals[2]) #m
        current_heading_ego = float(current_odo_vals[5]) #=yaw, rad
    # -- read prev odo file --
    with open(prev_odo,'r') as f2:
        prev_odo_vals=f2.read()
        prev_odo_vals=prev_odo_vals.split()
        prev_lat_ego = float(prev_odo_vals[0]) #deg
        prev_lon_ego = float(prev_odo_vals[1]) #deg
        prev_alt_ego = float(prev_odo_vals[2]) #m
        prev_heading_ego = float(prev_odo_vals[5]) #=yaw, rad

    current_track_ids = []
    current_tracks = []
    prev_track_ids = []
    prev_tracks = []
    # -- read track labels and append to consecutive frame lists --
    track_labels = csv.reader(open(track_labels_file), delimiter=" ")
    for label in track_labels:
        if int(label[0]) == int(test_id*2):
            current_track_ids.append(int(label[1]))
            current_tracks.append(label)
        if int(label[0]) == int((test_id*2)-1): 
            prev_track_ids.append(int(label[1]))
            prev_tracks.append(label)
    retained_ids = np.intersect1d(np.array(current_track_ids), np.array(prev_track_ids)) #ids in both frames 

    # -- create equivalent dictionaries --
    current_track_dict = {key_id: {'label_attribs': track_item} for key_id, track_item in zip(current_track_ids, current_tracks)}
    prev_track_dict =    {key_id: {'label_attribs': track_item} for key_id, track_item in zip(prev_track_ids, prev_tracks)}

    motion_statuses = [] 
    # -- low speed upper bound (for analysis purposes)--
    ls_up_kmph = 40
    ls_up = ls_up_kmph * 0.28 * timestep
    low_speeds = [] 

    # -- loop retained ids to get centers of moving vehicles --
    for key_id in retained_ids:
        # get prev track centers
        prev_center_cam = np.array([float(prev_track_dict[key_id]['label_attribs'][12]), float(prev_track_dict[key_id]['label_attribs'][13]), float(prev_track_dict[key_id]['label_attribs'][14])])
        prev_center_imu = cam_to_imu(prev_center_cam, T_ref0_ref2, T_velo_ref0, T_imu_velo)
        prev_geo_cords    = imu_to_geodetic(prev_center_imu[0], prev_center_imu[1], prev_center_imu[2], prev_lat_ego, prev_lon_ego, prev_alt_ego, prev_heading_ego)
        # get current track centers
        current_center_cam = np.array([float(current_track_dict[key_id]['label_attribs'][12]), float(current_track_dict[key_id]['label_attribs'][13]), float(current_track_dict[key_id]['label_attribs'][14])])
        current_center_imu = cam_to_imu(current_center_cam, T_ref0_ref2, T_velo_ref0, T_imu_velo)
        current_geo_cords    = imu_to_geodetic(current_center_imu[0], current_center_imu[1], current_center_imu[2], current_lat_ego, current_lon_ego, current_alt_ego, current_heading_ego)

        # -- Using geopy haversine to estimate distance travelled --
        distance_travelled = geopy.distance.geodesic((prev_geo_cords[0], prev_geo_cords[1]), (current_geo_cords[0], current_geo_cords[1])).meters

        # -- Condition to determine motion --
        if distance_travelled > dist_thresh:
            motion_statuses.append('m')
            if distance_travelled < ls_up:
                low_speeds.append('y') #speed between 10 and 30
            else:
                low_speeds.append('n') #speed greater than 30
        else:
            motion_statuses.append('s')
            low_speeds.append('n') # speed less than 10

    retained_dict = {retained_id: {'motion_status': motion_status, 'low_speed':low_speed} for retained_id, motion_status, low_speed in zip(retained_ids, motion_statuses, low_speeds)}
    return retained_dict, motion_statuses

def apply_canny_edge(region_i, mask_i):
    """
    Apply Canny edge detection to a given region

    Returns:
        img_blur: region with zero regions of mask applied
        edged_i_sum = the edge density value 
    """
    img_blur = region_i.copy()
    img_blur[mask_i==0]=0
    edged_i = cv2.Canny(image=img_blur, threshold1=60, threshold2=180) # Canny Edge Detection
    edged_i[mask_i==0] = 0 # ensure correct assignment of null pixels
    edged_i_sum = np.sum(edged_i==255) # edge density
    return img_blur, edged_i_sum

def corner_means(img_blur_i, corn_height, corn_width):
    """
    Get mean of all four defined corners within a region Canny edge detection to a given region

    Returns:
        a list of the four corner mean values 
    """
    # -- Top left corner (0) --
    tl_mean = int(np.mean(img_blur_i[:corn_height , :corn_width].astype(np.uint8)))
    # -- Top right corner (2) --
    tr_mean = int(np.mean(img_blur_i[:corn_height , -corn_width:].astype(np.uint8)))
    # -- Bottom left corner (1) --
    bl_mean = int(np.mean(img_blur_i[-corn_height: , :corn_width].astype(np.uint8)))
    # -- Bottom right corner (3) --
    br_mean = int(np.mean(img_blur_i[-corn_height: , -corn_width:].astype(np.uint8)))
    return [tl_mean, bl_mean, tr_mean, br_mean]

def max_intensity_check(dir_idx, abs_matrix, bg_pool_max, lid_pool_max, mask_bg, mask_fg):
    """
    Check if the maximum pixel intensity is equal in the background and foreground regions
    Args:
        dir_idx: channel (0, 1, or 2)
        abs_matrix: absolute value matrix corresponding to transformed scene flows 
        bg_pool_max, lid_pool_max: 2 arrays, each of size 3 corresponding to the pooled maximums
        mask_bg, mask_fg: background and neighbourhood masks

    Returns:
        dup_max: boolean value suggesting whether the maximum value of the two regions 
                 are the same or not
    """
    # -- Store pixels with max bg and max fg areas --
    bg_max_coords, fg_max_coords = [], []

    # -- background mask-based --
    idx_r_bg, idx_c_bg, _ = np.where(abs_matrix[dir_idx]==bg_pool_max[dir_idx])
    for count in range (len(idx_r_bg)):
        if mask_bg[idx_r_bg[count], idx_c_bg[count]]==255:
            bg_max_coords.append([idx_r_bg[count], idx_c_bg[count]])

    # -- neighbourhood mask-based --
    idx_r_fg, idx_c_fg, _ = np.where(abs_matrix[dir_idx]==lid_pool_max[dir_idx])
    for count in range (len(idx_r_fg)):
        if mask_fg[:,:,[0]][idx_r_fg[count], idx_c_fg[count]]==255:
            fg_max_coords.append([idx_r_fg[count], idx_c_fg[count]])

    # -- check if the max coords overlap --
    dup_max = False
    for coord_fg in fg_max_coords:
        for coord_bg in bg_max_coords:
            if coord_fg[0] == coord_bg[0]:
                if coord_fg[1] == coord_bg[1]:
                    dup_max = True
    return dup_max

def resolve_overlaps(image_rgb, kept_indexes, all_indexes):
    """
    Check if the maximum pixel intensity is equal in the background and foreground regions
    Args:
        image_rgb: RGB image of the left camera 
        kept_indexes: filtered frame indexes  
        all_indexes: all frame indexes

    Returns:
        motion_rois_overlap_dict: dictionary containing information on overlaps/V2V occlusions
                                  (relevant frame ids and intersection ratios as keys)
    """
    intersection_mask = np.zeros(image_rgb.shape, np.uint8)
    # -- Dictionary to store details of overlapping and moving vehicles --
    motion_rois_overlap_dict = {}
    # -- Loop for all regions --
    for count, label in enumerate(all_indexes):
        # -- Motion filtered regions in z direction
        if count in kept_indexes:
            current_coords = all_indexes[count]
            current_box    = [int(float(current_coords[1])), int(float(current_coords[2])), int(float(current_coords[3])), int(float(current_coords[4])) ]
            current_depth = abs(float(current_coords[13])) # get depth of vehicle (could be done in multiple ways)
            intersection_mask = cv2.rectangle(intersection_mask, (current_box[0],current_box[1]), 
                (current_box[2],current_box[3]), (127,127,127), 1)
            overlapping_ids = []
            overlapping_areas = []
            # -- Loop other motion regions --
            for i in kept_indexes:
                if (i != count):
                    comp_coords = all_indexes[i]
                    comp_box    = [int(float(comp_coords[1])), int(float(comp_coords[2])), int(float(comp_coords[3])), int(float(comp_coords[4]))]
                    comp_depth = abs(float(comp_coords[13])) # get depth of comparing vehicle (could be done in multiple ways)
                    # -- Calculate intersection area --
                    x_0 = np.max([current_box[0], comp_box[0]])
                    y_0 = np.max([current_box[1], comp_box[1]])
                    x_1 = np.min([current_box[2], comp_box[2]])
                    y_1 = np.min([current_box[3], comp_box[3]])
                    intersection_area_w = max(0, x_1 - x_0)
                    intersection_area_h = max(0, y_1 - y_0)
                    intersection_area = intersection_area_w * intersection_area_h
                    # -- Append to overlap dictionary on a conditional basis --
                    if intersection_area > 0 and current_depth > comp_depth:
                        intersection_mask = cv2.rectangle(intersection_mask, (current_box[0],current_box[1]), 
                            (current_box[2],current_box[3]), (255,255,255), 1)
                        intersection_mask = cv2.rectangle(intersection_mask, (x_0,y_0), 
                            (x_1,y_1), (0,0,255), -1)
                        # -- Append id --
                        overlapping_ids.append(i)
                        # -- Append ratio --
                        inter_ratio = intersection_area/( (int(float(current_coords[4]))-int(float(current_coords[2]))) * (int(float(current_coords[3]))-int(float(current_coords[1]))) )
                        overlapping_areas.append(inter_ratio)
            motion_rois_overlap_dict[count] = {'overlapping_ids': overlapping_ids, 'intersection_ratios': overlapping_areas}
    return motion_rois_overlap_dict

def move_grabcut(labels_txt, kept_indexes, ref_matrix, frame_id, bg_tolerance, 
                 bg_color, pr_bg_color, fg_color, pr_fg_color, line_thickness, 
                 f_p_scale, f_scale, gc_roi_dir):
    """
    Apply the tailored grabcut segmentation algorithm for the given region
    Args:
        labels_txt: path to annotated regions
        kept_indexes: filtered frame indexes  
        ref_matrix: input image for segmentation
        frame_id: frame id for current timestep
        bg_tolerance: tolerance for separation between regions and relative background
        bg_color, pr_bg_color: background and probable background hints 
        fg_color, pr_fg_color: foreground and probable foreground hints
        line_thickness: line thickness for generating the mask of hints  
        f_p_scale, f_scale: empirically obtained scaling factors for the probable foreground
            and foreground in generating the mask of hints  
        gc_roi_dir

    Returns:
        gc_imgs: grabcut images for the regions
        gc_masks: grabcut masks for the regions
    """
    # -- Lists to store grabcut ROIS and ROI masks
    gc_imgs, gc_masks = [], []
    gc_int_mask = np.zeros(ref_matrix.shape, np.uint8)

    labels = csv.reader(open(labels_txt), delimiter=" ")
    # -- Loop for all valid regions --
    for count, label in enumerate(labels):
        tmp_img = '%06d_10_%s.png'%(frame_id, count)  
        if count in kept_indexes:
            # -- Ratio increase (for background approximation) --
            # -- Calculate the amount to increase the region size --
            width  = int(float(label[3])) - int(float(label[1]))
            height = int(float(label[4])) - int(float(label[2]))
            increase_width  = int(width * bg_tolerance)
            increase_height = int(height * bg_tolerance)

            # -- Calculate dilated region coordinates --
            y_s = np.amax([int(float(label[2]))-increase_height,0])
            y_e = np.amin([int(float(label[4]))+increase_height,ref_matrix.shape[0]])
            x_s = np.amax([int(float(label[1]))-increase_width,0])
            x_e = np.amin([int(float(label[3]))+increase_width,ref_matrix.shape[1]])

            # -- Initialize temp mask for only current region --
            mask_filter = np.zeros(ref_matrix.shape, np.uint8)

            # -- Grabcut mask: set all to obvious background colour --
            mask_filter = cv2.rectangle(mask_filter, (x_s,y_s), (x_e,y_e), bg_color, line_thickness)
            gc_int_mask = cv2.rectangle(gc_int_mask, (x_s,y_s), (x_e,y_e), bg_color, line_thickness)
            
            # -- Grabcut mask: set all in original region to probable background colour 
            mask_filter = cv2.rectangle(mask_filter, (int(float(label[1])),int(float(label[2]))), (int(float(label[3])),int(float(label[4]))), pr_bg_color, line_thickness)
            gc_int_mask = cv2.rectangle(gc_int_mask, (int(float(label[1])),int(float(label[2]))), (int(float(label[3])),int(float(label[4]))), pr_bg_color, line_thickness)
            
            # -- Grabcut mask: hints by scaling down (set probable foreground colour) --
            diff_x = int(((int(float(label[3])) - int(float(label[1])))/f_p_scale)/2)
            diff_y = int(((int(float(label[4])) - int(float(label[2])))/f_p_scale)/2)
            mid_x = int((int(float(label[3])) + int(float(label[1])))/2)
            mid_y = int((int(float(label[2])) + int(float(label[4])))/2)
            yy_s = int(mid_y - diff_y)
            xx_s = int(mid_x - diff_x)
            yy_e = int(mid_y + diff_y)
            xx_e = int(mid_x + diff_x)
            mask_filter = cv2.rectangle(mask_filter, (xx_s,yy_s), (xx_e,yy_e), pr_fg_color, line_thickness)
            gc_int_mask = cv2.rectangle(gc_int_mask, (xx_s,yy_s), (xx_e,yy_e), pr_fg_color, line_thickness)

            # -- Check if roi edges are at the image edges--
            left_edge = False
            right_edge = False
            top_edge = False
            btm_edge = False
            if int(x_s)<increase_width:
                left_edge = True
            if int(x_e)>=(int(ref_matrix.shape[1])-increase_width):
                right_edge = True
            if int(y_s)<increase_height:
                top_edge = True
            if int(y_e)>=(int(ref_matrix.shape[0])-increase_height):
                btm_edge = True

            # -- Set region at edges as probable foreground if needed --
            if top_edge:
                # -- Top edge --
                mask_filter = cv2.rectangle(mask_filter, (int(float(label[1])),int(float(label[2]))), (int(float(label[3])),yy_s), pr_fg_color, line_thickness)
                gc_int_mask = cv2.rectangle(gc_int_mask, (int(float(label[1])),int(float(label[2]))), (int(float(label[3])),yy_s), pr_fg_color, line_thickness)
            if right_edge:
                # -- Right edge --
                mask_filter = cv2.rectangle(mask_filter, (xx_e,int(float(label[2]))), (int(float(label[3])),int(float(label[4]))), pr_fg_color, line_thickness)
                gc_int_mask = cv2.rectangle(gc_int_mask, (xx_e,int(float(label[2]))), (int(float(label[3])),int(float(label[4]))), pr_fg_color, line_thickness)
            if btm_edge:
                # -- Bottom edge --
                mask_filter = cv2.rectangle(mask_filter, (int(float(label[1])),yy_e), (int(float(label[3])),int(float(label[4]))), pr_fg_color, line_thickness)
                gc_int_mask = cv2.rectangle(gc_int_mask, (int(float(label[1])),yy_e), (int(float(label[3])),int(float(label[4]))), pr_fg_color, line_thickness)
            if left_edge:
                # -- Left edge corner --
                mask_filter = cv2.rectangle(mask_filter, (int(float(label[1])),int(float(label[2]))), (xx_s,int(float(label[4]))), pr_fg_color, line_thickness)
                gc_int_mask = cv2.rectangle(gc_int_mask, (int(float(label[1])),int(float(label[2]))), (xx_s,int(float(label[4]))), pr_fg_color, line_thickness)

            # -- Grabcut mask: hints by scaling down (set obvious foreground colour) --
            diff_x = int(((int(float(label[3])) - int(float(label[1])))/f_scale)/2)
            diff_y = int(((int(float(label[4])) - int(float(label[2])))/f_scale)/2)
            mid_x = int((int(float(label[3])) + int(float(label[1])))/2)
            mid_y = int((int(float(label[2])) + int(float(label[4])))/2)
            yy_s = int(mid_y - diff_y)
            xx_s = int(mid_x - diff_x)
            yy_e = int(mid_y + diff_y)
            xx_e = int(mid_x + diff_x)
            mask_filter = cv2.rectangle(mask_filter, (xx_s,yy_s), (xx_e,yy_e), fg_color, line_thickness)
            gc_int_mask = cv2.rectangle(gc_int_mask, (xx_s,yy_s), (xx_e,yy_e), fg_color, line_thickness)

            # -- Load image to cut --
            img_grabcut = cv2.bitwise_and(ref_matrix,ref_matrix,mask = mask_filter)
            img_grabcut = cv2.cvtColor(img_grabcut,cv2.COLOR_GRAY2RGB)

            # -- Grabcut mask: GC values --
            mask_grabcut = np.zeros(img_grabcut.shape[:2], np.uint8)
            mask_grabcut[mask_filter==bg_color]  = cv2.GC_BGD
            mask_grabcut[mask_filter==pr_bg_color] = cv2.GC_PR_BGD
            mask_grabcut[mask_filter==pr_fg_color] = cv2.GC_PR_FGD
            mask_grabcut[mask_filter==fg_color] = cv2.GC_FGD

            # -- Foreground & background arrays --
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)

            # -- Grabcut algorithm --
            mask_grabcut, bgdModel, fgdModel = cv2.grabCut(img_grabcut,mask_grabcut,None,
                bgdModel,fgdModel,32,cv2.GC_INIT_WITH_MASK)
            mask_f1 = np.where((mask_grabcut==2)|(mask_grabcut==0),0,1).astype('uint8')
            img_grabcut = img_grabcut*mask_f1[:,:,np.newaxis]
            gc_masks.append(mask_f1) 
            gc_imgs.append(img_grabcut)

            # -- Save the grabcut region --
            cv2.imwrite('%s/%s' % (gc_roi_dir, tmp_img),img_grabcut)

    return gc_imgs, gc_masks

def yo_gc_fusion_spatial(fusion_roi_dir, current_gc, img_name, yolo_det_path, yolo_sum,
                        thresh_bb_sum, thresh_lb_sum, thresh_inter, lbl, motion_rois_overlap_dict,
                        index, seg_class, yolo_seg, gc_optim_mask, gc_optim_masks):
    """
    Apply the YOLO-GRABCUT fusion model for the spatial scene-flow representations
    Args:
        fusion_roi_dir: path to current region of grabcut segmentation 
        current_gc: grabcut image 
        img_name: name to save the image
        yolo_det_path: path to yolo result
        yolo_sum: sum of yolo mask pixels 
        thresh_bb_sum, thresh_lb_sum, thresh_lb_sum: empirically obtained fusion thresholds
        lbl: values defining the bounds of the region
        motion_rois_overlap_dict: dictionary containing information on overlaps/V2V occlusions
                                  (relevant frame ids and intersection ratios as keys)
        index: id of the current region to segment  
        seg_class: object class recognized by yolo 
        yolo_seg: yolo segmented region 
        gc_optim_mask: running image to add each segmented region post fusion
        gc_optim_masks: running list to append each segmented region post fusion 

    Returns:
        gc_optim_mask: running image to add each segmented region post fusion
        gc_optim_masks: running list to append each segmented region post fusion
    """
    gc_roi = current_gc[ int(float(lbl[2])):int(float(lbl[4])) , int(float(lbl[1])):int(float(lbl[3]))].astype(np.uint8)
    # -- Get sum of grabcut mask pixels and region total --
    sum_grabcut = np.sum(gc_roi!=0)
    sum_roi = gc_roi.shape[0] * gc_roi.shape[1]

    # -- Check if ROI has no segmentations by checking labels folder -- 
    if len(os.listdir(yolo_det_path))!=0:
        # -- Get ratio of yolo mask pixels w.r.t to total combined pixels --
        sum_ratio = yolo_sum/(yolo_sum+sum_grabcut)

        # -- Check fully oversegmented instances --
        yolo_not_overseg = True
        if (sum_roi < thresh_bb_sum): #small region
            if int(yolo_sum)==(sum_roi): #overseg   
                yolo_not_overseg = False

        # -- Get maximum intersection area of motion region and set threshold --
        motion_overlap_ratios = motion_rois_overlap_dict[index]['intersection_ratios']
        if len(motion_overlap_ratios)>0:
            inter_max = np.max(motion_overlap_ratios)
        else:
            inter_max = 0.0
            # -- Case 1: Assign yolo mask --
            #Conditions: Intersection ratio, ROI size, Yolo & GC mask comparative sizes, and Yolo mask w.r.t ROI                                            
            if (inter_max < thresh_inter) and (yolo_not_overseg) and (sum_ratio > thresh_lb_sum) and (seg_class):
                # -- Grabcut mask converted from grayscale to binary using yolo segmentation foreground --
                gc_roi[yolo_seg==0]=0
                gc_roi[yolo_seg>0]=255
            # -- Case 2: Failed condition ==> aassign grabcut mask i.e. do nothing
            else:
                pass
    else:
        # -- Case 3: Sum of yolo mask is 0 (no yolo detection) ==> assign grabcut mask i.e. do nothing --
        pass

    # -- Set new/optimized ROI --
    gc_optim_mask[ int(float(lbl[2])):int(float(lbl[4])) , int(float(lbl[1])):int(float(lbl[3])) ] = gc_roi.astype(np.uint8)
    gc_optim_masks.append(gc_optim_mask)
    # -- Save fusion in given direction --
    cv2.imwrite('%s/%s' % (fusion_roi_dir, img_name),gc_roi.astype(np.uint8))

    return gc_optim_masks, gc_roi

def viz_predictions(to_mask, img_rgb, mask_to_add):
    """
    Apply, if needed, an overlay of the neighbourhood mask for visualization
    Args:
        to_mask: apply mask if TRUE, else return rgb equivalent 
        img_rgb: RGB image  
        mask_to_add: 3-channel binary mask

    Returns:
        final_viz: image with/without the mask overlay
    """
    if to_mask:
        #darkened_rgb = (img_rgb*0.5).astype(np.uin8)
        #final_viz = cv2.addWeighted(darkened_rgb, 1, img_rgb, 1, 0)
        #final_viz = np.where(mask_to_add==0, darkened_rgb, img_rgb)
        # Define the factor by which to darken the background (e.g., 0.5 for 50% brightness)
        darken_factor = 0.75

        # Create a mask for the background
        background_mask = 1 - (mask_to_add / 255.0)  # Scale to [0, 1]

        # Create the darkened background by multiplying the RGB image with the mask
        final_viz = (img_rgb * (1 - darken_factor * background_mask)).astype(np.uint8)

        
    else:
        final_viz = img_rgb.copy()
    return final_viz

# ***************************** END: HELPER FUNCTIONS/WRAPPERS ***************************** 

class Evaluator:
    def __init__(self, device: torch.device, cfgs:DictConfig):
        self.cfgs = cfgs
        self.device = device

        logging.info(f' -- Loading test data from {self.cfgs.testset.root_dir} -- ')
        logging.info(f' -- Dataset split: {self.cfgs.testset.split} -- ')

        # -- Define test dataset path --
        ds_current = str(self.cfgs.testset.root_dir)[-4:]
        ds_sum_path = f'$PATH_TO_DATASETS_DIRECTORY/datasets/kitti_scene_flow_{ds_current}/testing/bbox_labels' #YOUR CODE: change path to your dataset annotations location
        ds_sum_txts = glob.glob(os.path.join(ds_sum_path, '*.txt'))
        ds_sum = (len(ds_sum_txts))
        self.test_dataset = KITTITest(self.cfgs.testset, ds_sum)

        # -- Define test dataset loader object --
        self.test_loader = torch.utils.data.DataLoader(
            dataset = self.test_dataset,
            batch_size = self.cfgs.model.batch_size,
            num_workers = self.cfgs.testset.n_workers,
            pin_memory = True
        )

        # -- Initialize the prediction model and transfer to device --
        logging.info(' -- Initializing prediction model -- ')
        self.model = model_factory(self.cfgs.model).to(device=self.device)

        # -- Load the weights file i.e. checkpoint --
        logging.info(f' -- Loading checkpoint from {self.cfgs.ckpt.path} -- ')
        checkpoint = torch.load(self.cfgs.ckpt.path, self.device)

        # -- Load checkpoint state dictionary --
        self.model.load_state_dict(checkpoint['state_dict'], strict=self.cfgs.ckpt.strict)

        # -- Get the MoVe parameters --
        with open(cfgs.move_params, 'r') as param_file:
            self.move_params = yaml.safe_load(param_file)

    def run(self):
        # -- Set model to evaluation mode --
        self.model.eval()

        # -- Define the results directory (and sub directories) --
        out_dir = self.cfgs.out_dir
        os.makedirs('%s/disp_0' % out_dir, exist_ok=True) #temp folder
        os.makedirs('%s/disp_c' % out_dir, exist_ok=True) #temp folder
        os.makedirs('%s/image_grids' % out_dir, exist_ok=True)
        os.makedirs('%s/temp_RGB_roi' % out_dir, exist_ok=True) #temp folder
        os.makedirs('%s/temp_YOLO_roi' % out_dir, exist_ok=True) #temp folder
        os.makedirs('%s/temp_gc_z_roi' % out_dir, exist_ok=True) #temp folder
        os.makedirs('%s/temp_gc_x_roi' % out_dir, exist_ok=True) #temp folder
        os.makedirs('%s/temp_z_fusion_roi' % out_dir, exist_ok=True) #temp folder
        os.makedirs('%s/temp_x_fusion_roi' % out_dir, exist_ok=True) #temp folder
        os.makedirs('%s/temp_t_fusion_roi' % out_dir, exist_ok=True) #temp folder
        os.makedirs('%s/temp_fusion_roi' % out_dir, exist_ok=True) #temp folder
        os.makedirs('%s/motion_predictions' % out_dir, exist_ok=True)
        os.makedirs('%s/fused_segmentations' % out_dir, exist_ok=True)
        os.makedirs('%s/motion_flow_matrices' % out_dir, exist_ok=True)


        # -- Define log file config (replace with a new one for each run) --
        log_out_txt = f'$PATH_TO_LOG_FILE/output_logs/log_ds_{args.ds}.txt' #YOUR CODE: Give absolute path to log file
        if os.path.exists(log_out_txt):
            os.remove(log_out_txt)
        new_log = open(log_out_txt, 'w')
        new_log.close()
        logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S    ',
                    filename=str(log_out_txt),
                    filemode='a')
        logger = logging.getLogger()

        # -- Create a file handler to redirect logging output to a file --
        file_handler = logging.FileHandler(log_out_txt)
        logger.addHandler(file_handler)

        # -- Define results file --
        results_out = f'results_annots/our_results_ds_{args.ds}.txt'
        if os.path.exists(results_out):
            os.remove(results_out)
        new_out = open(results_out, 'w')
        new_out.close()

        # -- Loop through inputs i.e. data loader --
        c = 0
        for inputs in tqdm(self.test_loader):
            logging.info(f' --------- Processing {c} ({args.ds}) ---------')
            inputs = utils.copy_to_device(inputs, self.device)

            if (c > 0): # eliminate the first pair 
                # -- Execute the model for scene-flow vector generation --
                with torch.cuda.amp.autocast(enabled=False):
                    outputs = self.model.forward(inputs)

                # -- Loop through each batch id from 0 to batch_size-1 and get camera intrinsics + odometry files --
                tempshape = (outputs['flow_2d']).shape[0]
                for batch_id in range(tempshape):
                    test_id = inputs['index'][batch_id].item() 
                    input_h = inputs['input_h'][batch_id].item() 
                    input_w = inputs['input_w'][batch_id].item() 
                    f  = inputs['intrinsics'][batch_id][0].item()   
                    cx = inputs['intrinsics'][batch_id][1].item()  
                    cy = inputs['intrinsics'][batch_id][2].item()  
                    odo_vals = '%s/%s/%s/%06d_10.txt' % (self.cfgs.testset.root_dir, 
                                                         'testing', 'oxts', test_id)
                    prev_odo_vals = '%s/%s/%s/%06d_11.txt' % (self.cfgs.testset.root_dir, 
                                                              'testing', 'oxts', int(test_id-1))

                    # -- Set params for motion prediction --
                    timestamps_path = '%s/%s/timestamps.txt' % (self.cfgs.testset.root_dir, 'testing')
                    cam2_total_seconds = timestamps2seconds(timestamps_path)
                    dt = np.median(np.diff(cam2_total_seconds))
                    lb_dist_thresh = float(self.move_params['ego_car']['min_move_spd']) 
                    lb_dist_thresh = lb_dist_thresh * 0.28 #ms-1
                    dist_threshold = dt * lb_dist_thresh

                    # -- Print frame id of current timestep --
                    logging.info(f' Frame ID: {str(test_id)} (ds: ({args.ds})) ')

                    if str(test_id) != '-1': # update line if you wish to test a single frame (i.e timestamp) 
                        # -- Read annotations file for the current timestep --
                        labels_file='%s/%s/%s/%06d_10.txt' % (self.cfgs.testset.root_dir, 
                                                              'testing', 'bbox_labels', test_id)

                        # -- Read forward speed and steer values --
                        speed_ms = preprocess_speed(odo_vals, 8)
                        speed_kmph = speed_ms*3.6
                        steer = preprocess_speed(odo_vals, 19)

                        # -- Define state of ego vehicle (driving scenario) --
                        ego_state = 'Regular'
                        steer_thresh   = float(self.move_params['ego_car']['max_steer_turn']) 
                        fwd_vel_thresh = float(self.move_params['ego_car']['max_stationary_spd'])  
                        if steer > steer_thresh:
                            ego_state = 'Turning'
                        elif speed_kmph < fwd_vel_thresh:
                            ego_state = 'Stopped'
                        
                        # -- Read original input image (left-front camera) --
                        rgb_path = '%s/%s/%s/%06d_10.png' % (self.cfgs.testset.root_dir, 
                                                             'testing', 'image_2', test_id)
                        rgb_img = cv2.imread(rgb_path) 


                        # ***************************** START: POINT CLOUD FILTERING ***************************** 
                        
                        # -- Filtering Step 1: Obtain LiDAR point cloud for original left camera image --
                        velodyn_path = '%s/%s/%s/%06d_10.bin' % (self.cfgs.testset.root_dir, 
                                                                 'testing', 'velodyne_points', test_id)
                        ptcloud  = np.fromfile(velodyn_path, dtype = np.float32).reshape(-1, 4)
                        
                        # -- Filtering Step 2: Transform LiDAR point cloud to left camera frame --
                        # -- Read calibration files --
                        data_dir_calib = f'$PATH_TO_CALIBRATION_FILES/calib_ds/training/{args.ds}' #YOUR CODE: update path
                        calib_imu_velo = os.path.join(data_dir_calib, 'calib_imu_to_velo.txt')
                        calib_velo_cam = os.path.join(data_dir_calib, 'calib_velo_to_cam.txt')
                        calib_cam_cam  = os.path.join(data_dir_calib, 'calib_cam_to_cam.txt')
                        calib_cam=read_calib_to_csv(calib_cam_cam, ' ')
                        calib_velo=read_calib_to_csv(calib_velo_cam, ' ')

                        # -- Get rotation matrix --
                        rot = np.array(calib_velo.loc['R:'], dtype=float)
                        rot = rot[np.isfinite(rot)].reshape((3,3))

                        # -- Get translation matrix --
                        trans = np.array(calib_velo.loc['T:'], dtype=float)
                        trans = trans[np.isfinite(trans)]

                        # -- Get transformation matrix --
                        tr = np.empty((3,4))
                        tr[:3, :3] = rot
                        tr[:3, 3] = trans

                        # -- Calibration of left cam --
                        P0 = np.array(calib_cam.loc['P_rect_00:'], dtype=float)
                        P0 = P0[np.isfinite(P0)].reshape((3,4))

                        # -- Map point cloud coordinates to left camera frame --
                        depth_limit = int(self.move_params['ego_car']['depth_cloud_img'])
                        cloud_img = pointcloud_to_image(ptcloud, input_w, input_h, 
                                                        tr, P0, depth_limit, labels_file)

                        # -- Get calibration matrices for tracks --
                        with open(calib_cam_cam,'r') as f1:
                            calib_tracks = f1.readlines()
                        
                        # -- Get rigid transformation from Camera 0 (ref) to Camera 2 --
                        R_2 = np.array([float(x) for x in calib_tracks[21].strip().split(' ')[1:]]).reshape((3,3))
                        t_2 = np.array([float(x) for x in calib_tracks[22].strip().split(' ')[1:]]).reshape((3,1))

                        # -- Get cam0 to cam2 rigid body transformation in homogeneous coordinates --
                        T_ref0_ref2 = np.insert(np.hstack((R_2, t_2)), 3, values=[0,0,0,1], axis=0)
                        T_velo_ref0 = get_rigid_transformation(calib_velo_cam)
                        T_imu_velo = get_rigid_transformation(calib_imu_velo)

                        # -- Filtering Step 3: Pixel filtering for valid depths with respect to point cloud --
                        filtered_img_lidar  = np.zeros((input_h, input_w, 3), dtype=np.uint8)
                        filtered_img_lidar[:,:]=(0, 0, 0)
                        
                        # -- Disregard all points whose depth is greater than the limit --
                        img_heights, img_widths = np.where(cloud_img < depth_limit) 
                        img_hw = np.array( list(zip(img_heights, img_widths)) )
                        cloud_img_rect = cloud_img[ img_hw[:, 0], img_hw[:, 1] ] #.reshape(-1,1)
                        
                        # -- Pixel filter based on lidar --
                        temp_count = 0
                        for count in range(cloud_img_rect.shape[0]):
                            temp_count = temp_count + 1
                            h = img_hw[count][0]
                            w = img_hw[count][1]
                            if (  (cloud_img_rect[count] == depth_limit) ):
                                logging.warning('-- CLOUD IMAGE WITH MAX DEPTH FOUND !!!!!!!!!! --')
                                continue #filtered_img_lidar[h, w] = (255, 255, 255)
                            filtered_img_lidar[h, w] = (255, 255, 255)
                        
                        # ***************************** END: POINT CLOUD FILTERING ***************************** 


                        # ***************************** START: GENERATE SCENE-FLOW MATRIX ***************************** 
                        
                        # -- Retrive 2D flow vectors, which are needed for generating the course occlusion mask --
                        flow_2d_pred = outputs['flow_2d'][batch_id] # e.g. shape: 2,376,1242
                        # -- Reorder elements of the error and keep all values within the valid range --
                        flow_2d_pred = flow_2d_pred.permute(1, 2, 0).clamp(-500, 500).cpu().detach().numpy()
                        flow_2d_pred = flow_2d_pred[:input_h, :input_w]

                        # -- disp_0: from GA-Net --
                        disp_0 = utils.load_disp_png(os.path.join(
                            self.test_dataset.root_dir, 'disp_ganet', '%06d_10.png' % test_id)
                        )[0]

                        # --  Retrive 3D flow vectors --
                        flow_3d_pred = outputs['flow_3d'][batch_id] # shape 3,8192

                        # -- Densification i.e. reconstructing using 3D flow vectors --
                        pc1_dense = utils.disp2pc(disp_0, baseline=0.54, f=f, cx=cx, cy=cy) 
                        pc1_dense = torch.from_numpy(pc1_dense.reshape([-1, 3]).transpose()).to(self.device)
                        pc1 = inputs['pcs'][batch_id, :3]
                        flow_3d_dense = knn_interpolation(
                            input_xyz=pc1[None, ...],
                            input_features=flow_3d_pred[None, ...],
                            query_xyz=pc1_dense[None, ...],
                        )[0]

                        # -- Compute disparity change --
                        pc1_dense_warp = pc1_dense + flow_3d_dense
                        disp_c = 0.54 * f / pc1_dense_warp[2].cpu().detach().numpy().reshape(input_h, input_w)
                        disp_c[disp_0 < 0] = -1.0

                        # -- Save disparities as grayscale images (temporarily)--
                        utils.save_disp_png('%s/disp_0/%06d_10.png' % (out_dir, test_id), disp_0)
                        utils.save_disp_png('%s/disp_c/%06d_10.png' % (out_dir, test_id), disp_c)

                        # -- Generate a coarse occlusion mask for rigid background refinement --
                        grid = mesh_grid(1, input_h, input_w, device='cpu', channel_first=False)[0].numpy()
                        grid_warp = grid + flow_2d_pred
                        x_out = np.logical_or(grid_warp[..., 0] < 0, grid_warp[..., 0] > input_w)
                        y_out = np.logical_or(grid_warp[..., 1] < 0, grid_warp[..., 1] > input_h)
                        occ_mask1_2d = np.logical_or(x_out, y_out).astype(np.uint8) * 255

                        # -- Ensure that the scene-flow matrix is coherent with the image shape --
                        flow_3d_dense = flow_3d_dense.permute(1, 0).cpu().detach().numpy()
                        flow_3d_dense_proj = flow_3d_dense.reshape(input_h, input_w, 3)
                        flow_3d_dense_proj_dup = flow_3d_dense_proj.copy()

                        # ***************************** END: GENERATE SCENE-FLOW MATRIX ***************************** 

                        # ***************************** START: CREATE NEIGHBOURHOOD MASK ***************************** 

                        # -- Dilate with kernel to fill foreground (use median filter to complete fill) --
                        kern_size_dilate = int(self.move_params['pt_cloud_masks']['kern_dilate'])
                        kernel = np.ones((kern_size_dilate,kern_size_dilate),np.uint8) 
                        lidar_filter = cv2.dilate(filtered_img_lidar,kernel,iterations = 3)
                        kern_size_blur = int(self.move_params['pt_cloud_masks']['kern_blur'])
                        for count in range(5):
                            lidar_filter = cv2.medianBlur(lidar_filter,kern_size_blur) 

                        # -- Take only one channel from lidar mask --
                        mask_lidar = np.squeeze(lidar_filter[:,:,[0]])

                        # ***************************** END: CREATE NEIGHBOURHOOD MASK ***************************** 

                        # ***************************** START: FILTER REGIONS WHICH ARE OUT OF THE NEIGHBOURHOOD MASK ***************************** 

                        # -- List to store pixel count inside bounding box and respective mean --
                        out_of_lidar_bb_ids = []
                        color = (255,255,255)
                        thickness = -1 #fill
                        valid_label_count = 0

                        # -- Read each region and apply mask --
                        mask_bbox = np.zeros(rgb_img.shape[:2],np.uint8)
                        labels = csv.reader(open(labels_file), delimiter=" ")
                        for bb_c, label in enumerate(labels):
                            valid_label_count = valid_label_count + 1
                            # -- Draw a white rectangle over bounding box mask to indicate foreground --
                            mask_bbox = cv2.rectangle(mask_bbox, (int(float(label[1])),int(float(label[2]))), (int(float(label[3])),int(float(label[4]))), color, thickness)
                            # -- Obtain current region and total number of pixels --
                            temp = mask_bbox[int(float(label[2])):int(float(label[4])),int(float(label[1])):int(float(label[3]))]
                            pix_no_bb = np.sum(temp!=-1)
                            # -- Get sum of pixels in lidar mask filter within the bounding box (USE FILLED LIDAR)--
                            temp_lf = mask_lidar[int(float(label[2])):int(float(label[4])),int(float(label[1])):int(float(label[3]))]
                            sum_fg_lidar = np.sum(temp_lf!=0)
                            # -- Threshold and determing if region is inside LiDAR region --
                            per_thresh = float(self.move_params['pt_cloud_masks']['min_region_inc_ratio']) 
                            sum_fg_lidar_rat = sum_fg_lidar/pix_no_bb
                            # -- Fill region in LiDAR mask if condition is true --
                            if sum_fg_lidar_rat>per_thresh:
                                lidar_filter = cv2.rectangle(lidar_filter, (int(float(label[1])),int(float(label[2]))), (int(float(label[3])),int(float(label[4]))), color, thickness)
                            else:
                                out_of_lidar_bb_ids.append(bb_c)

                        # ***************************** END: FILTER REGIONS WHICH ARE OUT OF THE NEIGHBOURHOOD MASK ***************************** 

                        # ***************************** START: CREATE NEIGHBOURHOOD MASK ***************************** 

                        # -- Scale up bounding box masks so that there is a good seperation between background and ROIs --
                        scale_bb = int(self.move_params['pt_cloud_masks']['kern_bb_up']) 
                        kernel = np.ones((scale_bb,scale_bb),np.uint8)                        
                        mask_bbox_dilated = cv2.dilate(mask_bbox,kernel,iterations = 1)
                        mask_bbox_inverse = cv2.bitwise_not(mask_bbox)

                        # -- Get the combined mask (actual/sure background within lidar region ignoring probable static road users) --
                        mask_bbox_inverse = cv2.bitwise_not(mask_bbox_dilated)
                        mask_combined_bg = cv2.bitwise_and(mask_lidar,mask_bbox_inverse)

                        # -- Apply course occlusion mask --
                        mask_combined_bg[occ_mask1_2d==255] = 0

                        # ***************************** END: CREATE NEIGHBOURHOOD MASK ***************************** 

                        channels = [flow_3d_dense_proj_dup[:,:,[0]].copy(), flow_3d_dense_proj_dup[:,:,[1]].copy(), flow_3d_dense_proj_dup[:,:,[2]].copy()]
                        channel_shifts = channels.copy()

                        # -- Obtain the means of the background region channel-wise (neglects probable background inside bounding boxes) --
                        # -- channel 1 --
                        img_mask_mean = channels[0][np.where(mask_combined_bg == 255)]
                        mean_c1 = np.mean(img_mask_mean, axis=0)[0]
                        # -- channel 2 --
                        img_mask_mean = channels[1][np.where(mask_combined_bg == 255)]
                        mean_c2 = np.mean(img_mask_mean, axis=0)[0]
                        # -- channel 3 --
                        img_mask_mean =  channels[2][np.where(mask_combined_bg == 255)]
                        mean_c3 = np.mean(img_mask_mean, axis=0)[0]
                        bg_avg_channel = [mean_c1, mean_c2, mean_c3]

                        # -- Shift each channel based on background mask and driving scenario --
                        # -- c2 not used as changes in the y direction between 2 consecutive frames is negligible --
                        baseline_left_m = float(self.move_params['channel_shift']['m'])  #NOTE:empirically obtained
                        baseline_left_ycept = float(self.move_params['channel_shift']['y_cept']) #NOTE:empirically obtained 
                        bg_avg_c1 = (steer * baseline_left_m) + baseline_left_ycept
                        if ego_state=='Stopped':
                            c1_shift_org = channel_shifts [0]
                            c2_shift_org = channel_shifts [1]
                            c3_shift_org = channel_shifts [2]
                        elif ego_state=='Turning': #discarded scenario
                            c1_shift_org = np.add(channel_shifts [0],-(bg_avg_channel[0]))
                            c2_shift_org = np.add(channel_shifts [1],-(bg_avg_channel[1]))
                            c3_shift_org = np.add(channel_shifts [2],-(bg_avg_channel[2]))
                        else:
                            c1_shift_org = np.add(channel_shifts [0],-bg_avg_c1) #use best fit line
                            c2_shift_org = np.add(channel_shifts [1],-(bg_avg_channel[1]))
                            c3_shift_org = np.add(channel_shifts [2],-(bg_avg_channel[2]))

                        # -- Take absolute values to intensify all regions of motion irrespective of the direction --                            
                        c1_abs = np.abs(c1_shift_org)
                        c2_abs = np.abs(c2_shift_org)
                        c3_abs = np.abs(c3_shift_org)                      
                        c1_abs_test = c1_abs.copy()
                        c2_abs_test = c2_abs.copy()
                        c3_abs_test = c3_abs.copy()
                        c_abs_tests = [c1_abs_test, c2_abs_test, c3_abs_test]
        
                        # -- Call custom pooling function -- 
                        kern_size = int(self.move_params['pooler']['kern'])
                        stride_size = int(self.move_params['pooler']['stride'])
                        c1_abs_test[np.where(np.squeeze(lidar_filter[:,:,[0]])==0)] = 0
                        max_c1_abs_bg, _, max_c1_abs_lid, _  = pool_max(c1_abs_test, name = 'c1',kern = kern_size, stride = stride_size, 
                                                              mask_bg = mask_combined_bg, mask_lid = lidar_filter[:,:,[0]])
                        c2_abs_test[np.where(np.squeeze(lidar_filter[:,:,[0]])==0)] = 0
                        max_c2_abs_bg, _, max_c2_abs_lid, _ = pool_max(c2_abs_test, name = 'c2',kern = kern_size, stride = stride_size, 
                                                              mask_bg = mask_combined_bg, mask_lid = lidar_filter[:,:,[0]])
                        c3_abs_test[np.where(np.squeeze(lidar_filter[:,:,[0]])==0)] = 0
                        max_c3_abs_bg, _, max_c3_abs_lid, _ = pool_max(c3_abs_test, name = 'c3',kern = kern_size, stride = stride_size, 
                                                              mask_bg = mask_combined_bg, mask_lid = lidar_filter[:,:,[0]])
                        pooled_bg_maxs = [max_c1_abs_bg, max_c2_abs_bg, max_c3_abs_bg]
                        pooled_lid_maxs = [max_c1_abs_lid, max_c2_abs_lid, max_c3_abs_lid]

                        # -- Show pixels with max bg and max fg areas (z and x directions) --
                        dup_flag_z = max_intensity_check(2, c_abs_tests, pooled_bg_maxs, pooled_lid_maxs, mask_combined_bg, lidar_filter)
                        dup_flag_x = max_intensity_check(0, c_abs_tests, pooled_bg_maxs, pooled_lid_maxs, mask_combined_bg, lidar_filter)

                        # -- Resolve flow region uncertainties (z and x directions) --
                        speed_thresh = int(self.move_params['flow_tol']['spd'])  
                        flow_thresh = float(self.move_params['flow_tol']['flow'])   
                        if dup_flag_z:
                            logging.warning('DUPLICATE FOUND (z)!!')
                            whole_max_lid_c3 = speed_thresh
                        else:
                            whole_max_lid_c3 = np.max([pooled_lid_maxs[2], flow_thresh])
                        if dup_flag_x:
                            logging.warning('DUPLICATE FOUND (x)!!')
                            whole_max_lid_c1 = speed_thresh
                        else:
                            whole_max_lid_c1 = np.max([pooled_lid_maxs[0], flow_thresh])
                        flow_3d_dense_proj_dup[np.where(np.squeeze(lidar_filter[:,:,[0]])==0)] = 0

                        # -- x (lateral) direction --
                        flow_3d_dense_proj_dup[:,:,[0]] = np.interp(c1_abs, (0, whole_max_lid_c1),(0,255))
                        flow_3d_dense_proj_dup[:,:,[0]] = flow_3d_dense_proj_dup[:,:,[0]].astype(np.uint8)

                        # -- y (vertical) direction --
                        flow_3d_dense_proj_dup[:,:,[1]] = np.zeros_like(flow_3d_dense_proj_dup[:,:,[1]]) 
                        flow_3d_dense_proj_dup[:,:,[1]] = flow_3d_dense_proj_dup[:,:,[1]].astype(np.uint8)

                        # -- z (longitudinal) direction --
                        flow_3d_dense_proj_dup[:,:,[2]] = np.interp(c3_abs, (0, whole_max_lid_c3),(0,255))
                        flow_3d_dense_proj_dup[:,:,[2]] = flow_3d_dense_proj_dup[:,:,[2]].astype(np.uint8)
                        
                        flow_3d_dense_proj_dup[np.where(np.squeeze(lidar_filter[:,:,[0]])==0)] = 0 # ensure mask-based region selection

                        # -- Save final scene flow-based representation --
                        tmpgray_z = flow_3d_dense_proj_dup[:,:,[2]].copy().squeeze()
                        tmpgray_x = flow_3d_dense_proj_dup[:,:,[0]].copy().squeeze()
                        final_res_z = tmpgray_z.astype(np.uint8)  
                        final_res_x = tmpgray_x.astype(np.uint8) 

                        # ***************************** START: VISUALIZE GRID OF VEHICLES WITH CORRESPONDING IDS ***************************** 

                        # -- List to store each image region in image intensity range of rgb image --
                        dist_imgs_rgb = []
                        
                        # -- Obtain image grid width and weight depending on the number of regions --
                        grid_txt_pos = []
                        grid_h = rgb_img.shape[0]
                        max_h = int(grid_h/2)
                        grid_w = rgb_img.shape[1]
                        if valid_label_count!=0:
                            max_w = int(grid_w/np.ceil(valid_label_count/2))
                        else:
                            max_w = 0

                        labels = csv.reader(open(labels_file), delimiter=" ")
                        # -- Loop and append all regions --
                        for label in labels:
                            # -- Select current region from rgb image --
                            interp_img = rgb_img[int(float(label[2])):int(float(label[4])),int(float(label[1])):int(float(label[3]))]
                            interp_img = interp_img.astype(np.uint8)

                            # -- Resize according to img grid slot dim --
                            resized_interp = cv2.resize(interp_img, (max_w, max_h), interpolation = cv2.INTER_AREA)

                            #dist_imgs.append(temp_img)
                            dist_imgs_rgb.append(resized_interp)
                        
                        # -- Fill image grid if there is at least 1 valid region --
                        img_grid = np.zeros((grid_h,grid_w,3),np.uint8)
                        if dist_imgs_rgb:
                            # -- Font scale for the text overlays --
                            f_scale_grid = 2 
                            # -- Variable to determine vertical position of text overlays --
                            f_pos_tol = 50 
                            # -- Loop all regions --
                            for img_count, _ in enumerate(dist_imgs_rgb):
                                # -- First row of the image grid --
                                if img_count<(np.ceil(valid_label_count/2)):
                                    w_s = int(img_count*max_w)
                                    img_grid[0:max_h,w_s:(w_s+max_w)] = dist_imgs_rgb[img_count].astype(np.uint8)
                                    fg_pos = (w_s,0+f_pos_tol)
                                    grid_txt_pos.append(fg_pos)
                                # -- Second row of the image grid --
                                else:
                                    new_count = img_count - (np.ceil(valid_label_count/2))
                                    w_s = int((new_count*max_w))
                                    h_s = int(max_h)
                                    img_grid[h_s:h_s+max_h,w_s:w_s+max_w] = dist_imgs_rgb[img_count].astype(np.uint8)
                                    fg_pos = (w_s,h_s+f_pos_tol)
                                    grid_txt_pos.append(fg_pos)
                            
                            bg_pos = (int(grid_w-max_w),int(max_h)+50)
                            grid_txt_pos.append(bg_pos)
                            for count in range(len(grid_txt_pos)):
                                if (count == (len(grid_txt_pos)-1)):
                                    continue
                                else:
                                    img_grid = cv2.putText(
                                    img = img_grid,
                                    text = str(count),
                                    org = grid_txt_pos[count],
                                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                                    fontScale = f_scale_grid,
                                    color = (0,0,255),
                                    thickness = 2
                                    )

                            cv2.imwrite('%s/image_grids/%06d_10.png' % (out_dir, test_id), img_grid)

                        # ***************************** END: VISUALIZE GRID OF VEHICLES WITH CORRESPONDING IDS ***************************** 
        
                        # -- Get background mean and std from final grayscale post-processed image (z direction) --
                        img_mask_mean = final_res_z[np.where(mask_combined_bg == 255)]
                        mean_c_img_z = np.mean(np.mean(img_mask_mean, axis=0))
                        std_c_img_z = np.mean(np.std(img_mask_mean, axis=0))

                        # -- Get background mean and std from final grayscale post-processed image (x direction) --
                        img_mask_mean = final_res_x[np.where(mask_combined_bg == 255)]
                        mean_c_img_x = np.mean(np.mean(img_mask_mean, axis=0))
                        std_c_img_x = np.mean(np.std(img_mask_mean, axis=0))

                        # ***************************** START: ADAPTIVE THRESHOLDING ***************************** 
                        
                        labels = csv.reader(open(labels_file), delimiter=" ")
                        for count,label in enumerate(labels):
                            # -- Select ROI within bounding box --
                            fg_region_z = final_res_z[int(float(label[2])):int(float(label[4])),int(float(label[1])):int(float(label[3]))].astype(np.uint8)
                            fg_region_x = final_res_x[int(float(label[2])):int(float(label[4])),int(float(label[1])):int(float(label[3]))].astype(np.uint8)
                            # -- Apply thresholding to this region -- 
                            thresh_block = int(self.move_params['adaptive_thresh']['block'])
                            thresh_c = int(self.move_params['adaptive_thresh']['c'])
                            th_fg_z = cv2.adaptiveThreshold(fg_region_z,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,thresh_block,thresh_c)
                            th_fg_x = cv2.adaptiveThreshold(fg_region_x,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,thresh_block,thresh_c)
                            ## -- Thresh to zero --
                            thresh_fg_z = cv2.multiply(fg_region_z, th_fg_z)
                            thresh_fg_x = cv2.multiply(fg_region_x, th_fg_x)
                            # -- Set the applied threshold to the old region i.e. ROI --
                            final_res_z[int(float(label[2])):int(float(label[4])),int(float(label[1])):int(float(label[3]))] = thresh_fg_z
                            final_res_x[int(float(label[2])):int(float(label[4])),int(float(label[1])):int(float(label[3]))] = thresh_fg_x
                        
                        # ***************************** END: ADAPTIVE THRESHOLDING ***************************** 

                        # ***************************** START: SCENE-FLOW BASED MOTION FILTERS ***************************** 

                        # -- Flow-processed representations (z and x directions) --
                        selected_thresh_z = final_res_z.copy()
                        selected_thresh_x = final_res_x.copy()

                        # -- Store all bounding boxes (so that we minize the need to open using the csv reader) --
                        labels = csv.reader(open(labels_file), delimiter=" ")
                        idx_all = []
                        for count,label in enumerate(labels):
                            idx_all.append(label)

                        # -- Lists to store vehicles ids which are screened by the motion filters --
                        idx_keep_z, idx_keep_x = [], []

                        labels = csv.reader(open(labels_file), delimiter=" ")
                        # -- Loop all regions to resolve for overlaps/partial V2V occlusions--
                        for count,label in enumerate(labels):
                            # -- Get current region --   
                            roi_z = selected_thresh_z[ int(float(label[2])):int(float(label[4])) , 
                                                      int(float(label[1])):int(float(label[3]))].astype(np.uint8)
                            roi_x = selected_thresh_x[ int(float(label[2])):int(float(label[4])) , 
                                                      int(float(label[1])):int(float(label[3]))].astype(np.uint8)
                            
                            # -- Initially resolve for overlaps/V2V occlusions--
                            mask_bb_no_overlap_z = np.zeros(selected_thresh_z.shape, np.uint8)
                            mask_bb_no_overlap_x = np.zeros(selected_thresh_x.shape, np.uint8)

                            # -- Fill region of no overlap mask with white --
                            mask_bb_no_overlap_z = cv2.rectangle(mask_bb_no_overlap_z, (int(float(label[1])),int(float(label[2]))), 
                                (int(float(label[3])),int(float(label[4]))), 255, -1)
                            mask_bb_no_overlap_x = cv2.rectangle(mask_bb_no_overlap_x, (int(float(label[1])),int(float(label[2]))), 
                                (int(float(label[3])),int(float(label[4]))), 255, -1)
                            
                            # -- List to store overlapping region indexes --
                            overlap_idxs = []
                            # -- List to store overlapping region corner idxs --
                            overlap_corner_idx = []
                            # -- List to store labels of all road users IN FRONT OF current one --
                            overlaps_in_front = []
                            # -- Loop over all other regions to check for overlaps --
                            for i, l in enumerate(idx_all):
                                # -- Condition to not compare current region with itself --
                                if i!=count:
                                    # -- Check if current road user is behind the other --
                                    if abs(float(label[13])) > abs(float(l[13])):
                                        overlaps_in_front.append(l)
                                        # -- Current region --
                                        tmp_roi = [int(float(l[1])),int(float(l[2])) , int(float(l[3])),int(float(l[4]))]
                                        # -- Function returns null array if there is no overlap, else returns overlapping corners --
                                        corn_overlaps = isRectangleOverlap([ int(float(label[1])),int(float(label[2])),
                                                                            int(float(label[3])),int(float(label[4]))], tmp_roi)
                                        if corn_overlaps:
                                            # -- Add overlapped corner idxs to array (if that point is not already there) --
                                            # 0: Top left, 1: Bottom left, 2: Top right, 3: Bottom right
                                            for point in corn_overlaps:
                                                if not point in overlap_corner_idx: # works for multiple overlaps
                                                    overlap_corner_idx.append(point)
                                            # -- Eliminate mask values with overlaps (set those values in mask to 0) --
                                            overlap_idxs.append(i)
                                            mask_bb_no_overlap_z = cv2.rectangle(mask_bb_no_overlap_z, (int(float(l[1])),int(float(l[2]))), 
                                                (int(float(l[3])),int(float(l[4]))), 0, -1)
                                            mask_bb_no_overlap_x = cv2.rectangle(mask_bb_no_overlap_x, (int(float(l[1])),int(float(l[2]))), 
                                                (int(float(l[3])),int(float(l[4]))), 0, -1)

                            # -- Get current region from overlap mask --
                            mask_bb_no_overlap_roi_z = mask_bb_no_overlap_z[ int(float(label[2])):int(float(label[4])) , 
                                                                            int(float(label[1])):int(float(label[3]))].astype(np.uint8)
                            mask_bb_no_overlap_roi_x = mask_bb_no_overlap_x[ int(float(label[2])):int(float(label[4])) , 
                                                                            int(float(label[1])):int(float(label[3]))].astype(np.uint8)
                            
                            # -- Get no. of pixels valid within the region after removal of overlapping ones (z and x directions) --
                            no_overlap_count_z = len(roi_z[mask_bb_no_overlap_roi_z==255])
                            no_overlap_count_x = len(roi_x[mask_bb_no_overlap_roi_x==255])

                            # -- Condition to check if there are no applicable foregrounds (z and x directions) --
                            if len(roi_z[mask_bb_no_overlap_roi_z==255])==0:
                                mean_roi_z = 0
                            else:
                                # -- Get mean, std, and max from the regions which are not overlapping --
                                mean_roi_z = np.mean(roi_z[mask_bb_no_overlap_roi_z==255])
                            if len(roi_x[mask_bb_no_overlap_roi_x==255])==0:
                                mean_roi_x = 0
                            else:
                                # -- Get mean, std, max and min from the regions which are not overlapping --
                                mean_roi_x = np.mean(roi_x[mask_bb_no_overlap_roi_x==255])
                            
                            # -- Apply canny edge detection (z direction) --
                            img_blur_z, edge_sum_z = apply_canny_edge(roi_z, mask_bb_no_overlap_roi_z) 

                            # -- Apply canny edge detection (x direction) --
                            img_blur_x, edge_sum_x = apply_canny_edge(roi_x, mask_bb_no_overlap_roi_x) 

                            # -- Get corner dimensions --
                            corn_scale  = int(self.move_params['emp_filts']['corn_scale']) 
                            corn_size_h = int(roi_z.shape[0]/corn_scale)
                            corn_size_h = np.max([corn_size_h, 2])
                            corn_size_w = int(roi_z.shape[1]/corn_scale)
                            corn_size_w = np.max([corn_size_w, 2])

                            # -- Get means of all 4 corners --
                            all_corns_z = corner_means(img_blur_z, corn_size_h, corn_size_w)
                            all_corns_x = corner_means(img_blur_x, corn_size_h, corn_size_w)
                            filt_corns_z,  filt_corns_x= [], []
                            corn_mean_z, corn_mean_x = 0, 0
                            
                            # -- Check if truncated (the first if condition is optional, and has been included
                            #    only to speed up execution)--
                            trunc_corns = []
                            if int(label[6])!=0:
                                # -- Check which corners should NOT be considered --
                                top_f = False
                                btm_f = False
                                right_f = False
                                left_f = False
                                if int(float(label[1]))==0:
                                    left_f = True
                                if int(float(label[2]))==0:
                                    top_f = True
                                if int(float(label[3]))==int(rgb_img.shape[1]-1):
                                    right_f = True
                                if int(float(label[4]))==int(rgb_img.shape[0]-1):
                                    btm_f = True
                                # -- Top left corner --
                                if (top_f and left_f) or (btm_f and left_f) or (top_f and right_f) or (left_f) or (top_f):
                                    trunc_corns.append(0)
                                # -- Top right corner --
                                if (top_f and left_f) or (top_f and right_f) or (btm_f and right_f) or (top_f) or (right_f):
                                    trunc_corns.append(2)
                                # -- Bottom left corner --
                                if (top_f and left_f) or (btm_f and left_f) or (btm_f and right_f) or (left_f) or (btm_f):
                                    trunc_corns.append(1)
                                # -- Bottom right corner --
                                if (btm_f and left_f) or (top_f and right_f) or (btm_f and right_f) or (right_f) or (btm_f):
                                    trunc_corns.append(3)

                            # -- Check overlapping corns and truncated corners (z) --
                            for i, val in enumerate(all_corns_z):
                                if (i in overlap_corner_idx) or (i in trunc_corns):
                                    pass
                                else:
                                    filt_corns_z.append(val)

                            # -- Check if there are no filtered corners (z) --
                            if not filt_corns_z:
                                corn_mean_z = mean_c_img_z #np.min(all_corns)
                            else:
                                corn_mean_z = np.max([np.mean(filt_corns_z), mean_c_img_z]) # make sure that the corner mean is not too low

                            # -- Check overlapping corns and truncated corners (x) --
                            for i, val in enumerate(all_corns_x):
                                if (i in overlap_corner_idx) or (i in trunc_corns):
                                    pass
                                else:
                                    filt_corns_x.append(val)

                            # -- Check if there are no filtered corners (x) --
                            if not filt_corns_x:
                                corn_mean_x = mean_c_img_x #np.min(all_corns)
                            else:
                                corn_mean_x = np.max([np.mean(filt_corns_x), mean_c_img_x]) # make sure that the corner mean is not too low
                            
                            # -- Get canny edge ratios (z and x directions) --
                            if no_overlap_count_z==0:
                                edge_rat_z = 0
                            else:
                                edge_rat_z = edge_sum_z/no_overlap_count_z
                            if no_overlap_count_x==0:
                                edge_rat_x = 0
                            else:
                                edge_rat_x = edge_sum_x/no_overlap_count_x

                            # -- Define a center scale and obtain the mean and standard deviation of the definite foreground within the region (filter) --
                            scale_f = float(self.move_params['emp_filts']['scale_f']) #NOTE:empirically obtained 
                            diff_x = ((int(float(label[3])) - int(float(label[1])))/scale_f)/2
                            diff_y = ((int(float(label[4])) - int(float(label[2])))/scale_f)/2
                            mid_x = (int(float(label[3])) - int(float(label[1])))/2
                            mid_y = (int(float(label[4])) - int(float(label[2])))/2
                            y_s = int(mid_y - diff_y)
                            x_s = int(mid_x - diff_x)
                            y_e = int(mid_y + diff_y)
                            x_e = int(mid_x + diff_x)

                            # -- z direction mean and std --
                            fg_roi_z = roi_z[y_s:y_e, x_s:x_e].astype(np.uint8)
                            fg_roi_mean_uni_z = np.mean(fg_roi_z) 
                            fg_roi_std_uni_z  = np.std(fg_roi_z)

                            # -- x direction mean and std --
                            fg_roi_x = roi_x[y_s:y_e, x_s:x_e].astype(np.uint8)
                            fg_roi_mean_uni_x = np.mean(fg_roi_x) 
                            fg_roi_std_uni_x  = np.std(fg_roi_x)

                            # -- Call Gausian Mixture Model function if occluded  --
                            if int(label[5])==0: # not occluded
                                # -- z direction --
                                fg_roi_mean_z = fg_roi_mean_uni_z.copy()
                                fg_roi_std_z = fg_roi_std_uni_z.copy()
                                # -- x direction --
                                fg_roi_mean_x = fg_roi_mean_uni_x.copy()
                                fg_roi_std_x = fg_roi_std_uni_x.copy()
                            elif int(label[5])==1: # partially occluded
                                # -- z direction --
                                gmm_means_z, gmm_stds_z = gaussian_mixture_model(fg_roi_z)
                                fg_roi_mean_z = np.max(gmm_means_z)
                                gmm_idx_z = np.argmax(gmm_means_z)
                                fg_roi_std_z  = gmm_stds_z[gmm_idx_z]
                                # -- x direction --
                                gmm_means_x, gmm_stds_x = gaussian_mixture_model(fg_roi_x)
                                fg_roi_mean_x = np.max(gmm_means_x)
                                gmm_idx_x = np.argmax(gmm_means_x)
                                fg_roi_std_x  = gmm_stds_x[gmm_idx_x]
                            else: # heavily occluded
                                # -- z direction --
                                gmm_means_z, gmm_stds_z = gaussian_mixture_model(img_blur_z)
                                fg_roi_mean_z = np.max(gmm_means_z)
                                gmm_idx_z = np.argmax(gmm_means_z)
                                fg_roi_std_z  = gmm_stds_z[gmm_idx_z]
                                # -- x direction --
                                gmm_means_x, gmm_stds_x = gaussian_mixture_model(img_blur_x)
                                fg_roi_mean_x = np.max(gmm_means_x)
                                gmm_idx_x = np.argmax(gmm_means_x)
                                fg_roi_std_x  = gmm_stds_x[gmm_idx_x]                     
                            
                            # -- Check if its not a vehicle category ("person sitting" not included since our dataset has no such instances) --
                            outlier_ru = False
                            if (str(label[0]) == "Pedestrian:") or (str(label[0]) == "Cyclist:") or (str(label[0]) == "Misc:"):
                                outlier_ru = True

                            # -- Condition for edges --
                            # -- z direction --
                            edge_flag_z = False
                            if edge_rat_z == 0.0:
                                if not outlier_ru:
                                    edge_flag_z = True
                            # -- x direction --
                            edge_flag_x = False
                            if edge_rat_x == 0.0:
                                if not outlier_ru:
                                    edge_flag_x = True

                            # -- Corner mean check --
                            # -- z direction --
                            roi_intensity_flag_z = False
                            if (fg_roi_mean_z < corn_mean_z) or (mean_roi_z < corn_mean_z):
                                if not outlier_ru:
                                    roi_intensity_flag_z = True
                            # -- x direction --
                            roi_intensity_flag_x = False
                            if (fg_roi_mean_x < corn_mean_x) or (mean_roi_x < corn_mean_x):
                                if not outlier_ru:
                                    roi_intensity_flag_x = True

                            # -- Intensity check (within region) -- 
                            # -- z direction --
                            ratio_tol = float(self.move_params['emp_filts']['region_tol'])
                            tol = ratio_tol * fg_roi_std_z
                            edge_tol_flag_z = False
                            if (fg_roi_mean_z - tol) < corn_mean_z:
                                if not outlier_ru:
                                    edge_tol_flag_z = True
                            # -- x direction --
                            tol = ratio_tol * fg_roi_std_x
                            edge_tol_flag_x = False
                            if (fg_roi_mean_x - tol) < corn_mean_x:
                                if not outlier_ru:
                                    edge_tol_flag_x = True
                            
                            # -- Intensity check (w.r.t background) --
                            # -- z direction --
                            min_flag_z = False 
                            if (fg_roi_mean_z)<(mean_c_img_z + (std_c_img_z/4)) and (fg_roi_mean_z)>(mean_c_img_z - (std_c_img_z/4)): 
                                if not outlier_ru:
                                    min_flag_z = True
                            # -- x direction --
                            min_flag_x = False
                            if (fg_roi_mean_x)<(mean_c_img_x + (std_c_img_x/4)) and (fg_roi_mean_x)>(mean_c_img_x - (std_c_img_x/4)): 
                                if not outlier_ru:
                                    min_flag_x = True
                            
                            # -- Check to validate that there are no vehicles outside the neighbourhood mask --
                            out_of_region_flag = False
                            if count in out_of_lidar_bb_ids:
                                out_of_region_flag = True

                            # -- Eliminate if at least one of the conditions is true (z and x directions) --
                            if (min_flag_z) or (edge_flag_z) or (roi_intensity_flag_z) or (edge_tol_flag_z) or (out_of_region_flag) or (outlier_ru): 
                                pass
                            else:
                                idx_keep_z.append(count)
                            if (min_flag_x) or (edge_flag_x) or (roi_intensity_flag_x) or (edge_tol_flag_x) or (out_of_region_flag) or (outlier_ru): 
                                pass
                            else:
                                idx_keep_x.append(count)
              

                        # -- Make new mask of only filtered regions --
                        color = 255
                        thickness = -1
                        mask_init_filt_z = np.zeros(selected_thresh_z.shape, np.uint8)
                        mask_init_filt_x = np.zeros(selected_thresh_x.shape, np.uint8)
                        labels = csv.reader(open(labels_file), delimiter=" ")
                        for count, label in enumerate(labels):
                            if count in idx_keep_z:
                                mask_init_filt_z = cv2.rectangle(mask_init_filt_z, (int(float(label[1])),int(float(label[2]))), (int(float(label[3])),int(float(label[4]))), color, thickness)
                            if count in idx_keep_x:
                                mask_init_filt_x = cv2.rectangle(mask_init_filt_x, (int(float(label[1])),int(float(label[2]))), (int(float(label[3])),int(float(label[4]))), color, thickness)
                        
                        masked_rgb_z = cv2.bitwise_and(rgb_img,rgb_img,mask = mask_init_filt_z)
                        masked_rgb_x = cv2.bitwise_and(rgb_img,rgb_img,mask = mask_init_filt_x)

                        # -- Flag to activate if there are no moving vehicles (can be optionally used to classify results for analysis)--
                        zero_motion_flag = False
                        if ((np.sum(masked_rgb_z!=0)) == 0) and ((np.sum(masked_rgb_x!=0)) == 0):
                            zero_motion_flag = True

                        # -- Resolve overlaps/V2V occlusions for filtered regions in z and x directions -- 
                        motion_rois_overlap_dict_z = resolve_overlaps(rgb_img, idx_keep_z, idx_all)
                        motion_rois_overlap_dict_x = resolve_overlaps(rgb_img, idx_keep_x, idx_all)
                        
                        # ***************************** END: SCENE-FLOW BASED MOTION FILTERS ***************************** 
  
                        # ***************************** START: Combine scene-flow based regions with tracks to get final motion predictions ***************************** 
                        # -- Get generated track details (model executed separately) -- --
                        em_labels_frame = []
                        em_track_ids = []
                        em_tracks = []
                        em_labels_file = f'eagermot/combined/{args.ds}.txt'
                        em_labels = csv.reader(open(em_labels_file), delimiter=" ")
                        for count, label in enumerate(em_labels):
                            if int(label[0]) == int(test_id*2):
                                em_labels_frame.append(label)
                                em_track_ids.append(int(label[1]))
                                em_tracks.append(label)
                        
                        # -- Set region comparison dictionary params --
                        ids, bb_match_ratios_sorted, em_track_ids_sorted = [], [], []
                        mf_statuses = [] # x, z, b (both) or n(none)

                        # -- Loop all regions --
                        for count, label in enumerate (idx_all):
                            # -- append id --
                            ids.append(count)
                            # -- append motion-filter statuses from scene-flows --
                            if (count in idx_keep_x) and (count in idx_keep_z):
                                mf_statuses.append('b')
                            elif (count in idx_keep_x) and (count not in idx_keep_z):
                                mf_statuses.append('x')
                            elif (count not in idx_keep_x) and (count in idx_keep_z):
                                mf_statuses.append('z')
                            else:
                                mf_statuses.append('n')
                            # -- get overlap ratios --
                            current_frame_bb_rats = []
                            for em_l in em_labels_frame:
                                current_coords = [int(float(label[1])), int(float(label[2])), int(float(label[3])), int(float(label[4]))]
                                comp_coords = [int(float(em_l[5])), int(float(em_l[6])), int(float(em_l[7])), int(float(em_l[8]))]
                                current_frame_bb_rats.append(calculate_overlap_ratio(current_coords, comp_coords))
                            # -- append region ratios --
                            bb_match_ratios_sorted.append(np.sort(current_frame_bb_rats)[::-1]) # desc order of overlaps
                            sorted_idxs_desc = np.argsort(current_frame_bb_rats)[::-1] # desc order of args (indexes)
                            # -- track ids in descending order for current region --
                            current_em_track_ids_sorted = [] 
                            for sorted_idx in sorted_idxs_desc:
                                current_em_track_ids_sorted.append(em_track_ids[sorted_idx])
                            # -- append track ids --
                            em_track_ids_sorted.append(current_em_track_ids_sorted)
                            assert len(current_frame_bb_rats)==len(current_em_track_ids_sorted)
                        # -- Set assertions to determine that everything to be inserted to the dictionary is as expected --
                        assert len(ids)==len(bb_match_ratios_sorted)==len(mf_statuses)==len(em_track_ids_sorted)
                        bb_match_info_dict = {key_id: {'mf_statuses': mf_status, 'sorted_track_ids': em_track_id_sorted, 'sorted_bb_ratios':bb_match_ratio_sorted} for key_id, mf_status, em_track_id_sorted,bb_match_ratio_sorted  in zip(ids, mf_statuses, em_track_ids_sorted, bb_match_ratios_sorted)}

                        # -- Match the bounding boxes correctly to each index (hungarian matching) --
                        # -- set match threshold --
                        match_thresh = 0.70
                        # -- to store max ratio for each track --
                        best_ratio_idxs = np.full(len(em_track_ids), -100.00) # for dictionary 
                        best_ratios = best_ratio_idxs.copy()
                        move_out = False # flag to repeat until no non-negative duplicates 
                        assigned_ids = [] # stores all assigned ids (values should be unique)
                        assigned_track_ids = [] # stores all assigned track ids (values should be unique)
                        dyn_em_track_ids = np.array(em_track_ids.copy())

                        while not move_out:
                            max_ratios = [] # store max ratio for each track id
                            max_ratio_idxs = [] # stores the index of max ratio for each track id (could have duplicates)
                            for track_id in dyn_em_track_ids:
                                # -- get best ratio and correponding id --
                                max_ratio = -1
                                max_ratio_idx = -1
                                # -- loop each entry in dictionary --
                                for dict_count in range(len(bb_match_info_dict)):
                                    # -- make sure taken ids are not relooped --
                                    if dict_count not in assigned_ids:
                                        # -- get corresponding id and ratio (dont be fooled by plural 'indices'!) --
                                        indices = np.where(np.array(bb_match_info_dict[dict_count]['sorted_track_ids'])==track_id)[0]
                                        assert len(indices)>0
                                        current_max = bb_match_info_dict[dict_count]['sorted_bb_ratios'][indices[0]] # ratio
                                        if current_max > max_ratio:
                                            max_ratio = current_max
                                            max_ratio_idx = dict_count
                                # -- compare to threshold --
                                if max_ratio > match_thresh:
                                    max_ratios.append(max_ratio)
                                    max_ratio_idxs.append(max_ratio_idx)
                                else:
                                    max_ratios.append(-1) #no match 
                                    max_ratio_idxs.append(-1) # no match
                            assert len(dyn_em_track_ids)==len(max_ratios)==len(max_ratio_idxs)
                            max_ratio_idxs = np.array(max_ratio_idxs)
                            max_ratios = np.array(max_ratios)

                            # -- check for duplicates in max_ratio_idxs --
                            repeated_idxs = set()
                            duplicate_idxs = list(set(x for x in max_ratio_idxs if x in repeated_idxs or repeated_idxs.add(x)))
                            duplicate_idxs = np.array(duplicate_idxs)
                            duplicate_idxs = duplicate_idxs[duplicate_idxs >= 0] #remove negative values
                            # -- if no duplicates or no duplicates with positive idx --
                            if len(duplicate_idxs)==0:
                                # -- set all values accordingly --
                                for set_c, track_id in enumerate(dyn_em_track_ids):
                                    orig_index = np.where(em_track_ids == track_id)[0][0] #idx of track in orig array
                                    orig_index_2 = np.where(em_track_ids == track_id)[0][0]
                                    best_ratios[orig_index] = max_ratios[set_c].copy() #use idx to put corresponding best ratio
                                    best_ratio_idxs[orig_index_2] = max_ratio_idxs[set_c].copy() #use idx to put corresponding best ratio index
                                    assigned_ids.append(max_ratio_idxs[set_c]) # to make sure that same index is not repeated for multiple tracks
                                    assigned_track_ids.append(track_id) # to make sure that the same track is not updated twice
                                move_out = True
                            else:
                                # -- loop each duplicate --
                                for duplicate in duplicate_idxs:
                                    # -- assign id to best ratio track --
                                    temp_indices = np.where(max_ratio_idxs == duplicate)[0] # indices containing the duplicate id
                                    temp_ratios = max_ratios[temp_indices] # corresponding ratios
                                    best_ratio_arg = np.argmax(temp_ratios) # highest ratio index
                                    best_ratio = np.max(temp_ratios) # highest ratio value
                                    best_ratio_index = temp_indices[best_ratio_arg] # ratio index of orig track ids
                                    best_ratio_idxs[best_ratio_index] = duplicate
                                    best_ratios[best_ratio_index] = best_ratio
                                    assigned_track_ids.append(em_track_ids[best_ratio_index]) # to make sure that the same track is not updated twice
                                    assigned_ids.append(duplicate) # to make sure that same index is not repeated for multiple tracks
                                    dyn_em_track_ids = dyn_em_track_ids[dyn_em_track_ids!=(em_track_ids[best_ratio_index])]
                        non_assigned_ids = np.setdiff1d(np.array(ids), np.array(assigned_ids))
                        assert len(assigned_ids) == len(assigned_track_ids)

                        # -- Transfer non assigned tracks to correct list --
                        non_assigned_tracks = []
                        for i in range(len(assigned_track_ids)):
                            if int(assigned_ids[i]) < 0:
                                non_assigned_tracks.append(assigned_track_ids[i])
                        non_neg_assigned_track_ids = np.array(assigned_track_ids)[np.array(assigned_track_ids)>=0]
                        non_neg_assigned_ids = np.array(assigned_ids)[np.array(assigned_ids)>=0]
                        assert len(non_neg_assigned_track_ids) == len(set(non_neg_assigned_track_ids))
                        assert len(non_neg_assigned_ids) == len(set(non_neg_assigned_ids))

                        # -- Convert to numpy arrays --
                        assigned_ids = np.array(assigned_ids)
                        non_assigned_ids = np.array(non_assigned_ids)
                        assigned_track_ids = np.array(assigned_track_ids)
                        non_assigned_tracks = np.array(non_assigned_tracks)

                        # -- Get moving status from all tracks (return dictionary) --
                        track_moving_statuses, arr_motion_statuses = check_motion(odo_vals, prev_odo_vals, em_labels_file, test_id, dist_threshold, T_ref0_ref2, T_velo_ref0, T_imu_velo, dt)
                        arr_motion_statuses = np.array(arr_motion_statuses)
                        movers_no = len(arr_motion_statuses[arr_motion_statuses=='m']) 
                        current_track_keys = np.array(list(track_moving_statuses.keys()))
                        non_tracks = np.array(np.setdiff1d(assigned_track_ids, current_track_keys))  

                        # -- List to store final motion predictions --
                        final_motion_picks = [] 
                        # -- Get ego state --
                        logging.info('Ego vehicle state: %s '%ego_state)
                        for count, label in enumerate(idx_all):
                            if ego_state == 'Stopped':
                                # Completely occluded ones rely on LiDAR
                                if (int(label[5])==2) and (count not in non_assigned_ids):
                                    id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                    if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                        mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                        assert mt_status=='s' or mt_status=='m'
                                        if mt_status=='m':
                                            final_motion_picks.append(count)
                                else:
                                    # Resolve too small occluded ones if possible with track
                                    # Get region size
                                    ru_size = ( int(float(label[3])) - int(float(label[1])) ) * ( int(float(label[4])) - int(float(label[2])) )
                                    if (ru_size < 3000) and (count not in non_assigned_ids):
                                        id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                        if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                            mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                            assert mt_status=='s' or mt_status=='m'
                                            if mt_status=='m':
                                                final_motion_picks.append(count)
                                    else:
                                        if (count in idx_keep_z):
                                            final_motion_picks.append(count)
                                        elif (count not in idx_keep_z) and (count in idx_keep_x):
                                            # check if it is moving using track
                                            if count not in non_assigned_ids: # check if id has a match with tracks 
                                                id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                                if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                                    mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                                    assert mt_status=='s' or mt_status=='m'
                                                    if mt_status=='m':
                                                        final_motion_picks.append(count)
                                                else:
                                                    final_motion_picks.append(count)
                                            # if no match 
                                            else:
                                                final_motion_picks.append(count)
                                        # Low speed and m in tracks
                                        elif (count not in idx_keep_z) and (count not in idx_keep_x) and (count not in non_assigned_ids):
                                            id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                            if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                                mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                                ls_status = track_moving_statuses[assigned_track_ids[id_index]]['low_speed']
                                                assert mt_status=='s' or mt_status=='m'
                                                assert ls_status=='y' or ls_status=='n'
                                                if mt_status=='m' and ls_status=='y':
                                                    final_motion_picks.append(count)
                            elif ego_state == 'Regular':
                                # Completely occluded ones rely on LiDAR
                                if (int(label[5])==2) and (count not in non_assigned_ids):
                                    id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                    if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                        mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                        assert mt_status=='s' or mt_status=='m'
                                        if mt_status=='m':
                                            final_motion_picks.append(count)
                                elif (movers_no > 3) and (count not in non_assigned_ids):
                                    id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                    if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                        mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                        assert mt_status=='s' or mt_status=='m'
                                        if mt_status=='m':
                                            final_motion_picks.append(count)
                                else:
                                    # Get region size
                                    ru_size = ( int(float(label[3])) - int(float(label[1])) ) * ( int(float(label[4])) - int(float(label[2])) )
                                    # Resolve too small occluded ones if possible with track
                                    if (ru_size < 5000) and (count not in non_assigned_ids):
                                        id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                        if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                            mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                            assert mt_status=='s' or mt_status=='m'
                                            if mt_status=='m':
                                                final_motion_picks.append(count)
                                    else:
                                        if (count in idx_keep_z):
                                            # check if it is stationary in tracks 
                                            if count not in non_assigned_ids:
                                                id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                                if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                                    mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                                    assert mt_status=='s' or mt_status=='m'
                                                    if mt_status=='m':
                                                        final_motion_picks.append(count)
                                            else:
                                                final_motion_picks.append(count)
                                        # Low speed and m in tracks
                                        elif (count not in idx_keep_z) and (count not in idx_keep_x) and (count not in non_assigned_ids):
                                            id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                            if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                                mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                                ls_status = track_moving_statuses[assigned_track_ids[id_index]]['low_speed']
                                                assert mt_status=='s' or mt_status=='m'
                                                assert ls_status=='y' or ls_status=='n'
                                                if mt_status=='m' and ls_status=='y':
                                                    final_motion_picks.append(count)
                            elif ego_state == 'Turning':
                                # Completely occluded ones rely on LiDAR
                                if (int(label[5])==2) and (count not in non_assigned_ids):
                                    id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                    if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                        mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                        assert mt_status=='s' or mt_status=='m'
                                        if mt_status=='m':
                                            final_motion_picks.append(count)
                                else:
                                    # Get region size
                                    ru_size = ( int(float(label[3])) - int(float(label[1])) ) * ( int(float(label[4])) - int(float(label[2])) )
                                    # Resolve too small occluded ones if possible with track
                                    if (ru_size < 12000) and (count not in non_assigned_ids):
                                        id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                        if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                            mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                            assert mt_status=='s' or mt_status=='m'
                                            if mt_status=='m':
                                                final_motion_picks.append(count)
                                    else:
                                        # priority to tracks 
                                        if count not in non_assigned_ids:
                                            id_index = np.where(assigned_ids == count)[0][0] # get actual id (corresponding to text file)
                                            if (assigned_track_ids[id_index] not in non_assigned_tracks) and (assigned_track_ids[id_index] not in non_tracks): # make sure corresponding track is there too
                                                mt_status = track_moving_statuses[assigned_track_ids[id_index]]['motion_status']
                                                assert mt_status=='s' or mt_status=='m'
                                                if mt_status=='m':
                                                    final_motion_picks.append(count)
                                                else:
                                                    # check z status
                                                    if (count in idx_keep_z) and (count not in idx_keep_x):
                                                        final_motion_picks.append(count)
                                            else:
                                                if (count in idx_keep_z) and (count not in idx_keep_x):
                                                    final_motion_picks.append(count)
                                        else:
                                            if (count in idx_keep_z) and (count not in idx_keep_x):
                                                final_motion_picks.append(count)                                
                            else:
                                logging.warning(f'    !!!!!!!!! INVALID DRIVING SCENARIO !!!!!!!! ({count})')

                        logging.info(f' Region ids for segmentation: {final_motion_picks} ')
                        assert len(final_motion_picks) == len(set(final_motion_picks))                        

                        # ***************************** END: Combine scene-flow based regions with tracks to get final motion predictions ***************************** 

                        # ***************************** START: GRABCUT SEGMENTATION ***************************** 
                        # -- Initialize parameters --
                        bg_tol = float(self.move_params['grabcut']['bg_tol']) #NOTE:empirically obtained 
                        color_fg = 255 
                        color_bg = 40
                        color_pr_bg = 100 
                        color_pr_fg = 150
                        thickness = -1
                        # -- Scale factor for 'hint' (NOTE: empirically obtained) 
                        scale_f   = float(self.move_params['grabcut']['scale_f'])
                        scale_f_p = float(self.move_params['grabcut']['scale_f_p']) 
                        
                        # -- Directory to store all individual ROIs (z and x directions) --
                        gc_z_roi_dir = '%s/temp_gc_z_roi/%06d_10' % (out_dir, test_id)
                        os.makedirs(gc_z_roi_dir, exist_ok=True) 
                        gc_x_roi_dir = '%s/temp_gc_x_roi/%06d_10' % (out_dir, test_id)
                        os.makedirs(gc_x_roi_dir, exist_ok=True)

                        # -- Execute grabcut (z and x directions)--
                        gc_imgs_z, gc_masks_z  = move_grabcut(labels_file, idx_keep_z, selected_thresh_z, test_id, bg_tol,
                                                 color_bg, color_pr_bg, color_fg, color_pr_fg, thickness,
                                                 scale_f_p, scale_f, gc_z_roi_dir)
                        gc_imgs_x, gc_masks_x  = move_grabcut(labels_file, idx_keep_x, selected_thresh_x, test_id, bg_tol,
                                                 color_bg, color_pr_bg, color_fg, color_pr_fg, thickness,
                                                 scale_f_p, scale_f, gc_x_roi_dir)
                        
                        # -- Combine grabcut masks, combine grabcut images
                        # -- z direction --
                        os.makedirs('%s/mask_grabcut_z' % out_dir, exist_ok=True)
                        if len(gc_imgs_z)==0:
                            gc_img = np.zeros(rgb_img.shape, np.uint8)
                            gc_mask = np.zeros(rgb_img.shape, np.uint8)
                            cv2.imwrite('%s/mask_grabcut_z/%06d_10.png' % (out_dir, test_id),gc_mask)
                        elif len(gc_imgs_z)==1:
                            gc_img = gc_imgs_z[0]
                            gc_mask = gc_masks_z[0]
                            gc_mask_inter = np.interp(gc_mask, (0, 1),(0,255))
                            cv2.imwrite('%s/mask_grabcut_z/%06d_10.png' % (out_dir, test_id),gc_mask_inter)
                        else:
                            gc_img = gc_imgs_z[0]
                            gc_mask = gc_masks_z[0]
                            for img_count in range(1,len(gc_imgs_z)):
                                gc_img = cv2.bitwise_or(gc_img, gc_imgs_z[img_count])
                                gc_mask = cv2.bitwise_or(gc_mask, gc_masks_z[img_count])
                                gc_mask_inter = np.interp(gc_mask, (0, 1),(0,255))
                            cv2.imwrite('%s/mask_grabcut_z/%06d_10.png' % (out_dir, test_id),gc_mask_inter)

                        # -- x direction --
                        os.makedirs('%s/mask_grabcut_x' % out_dir, exist_ok=True)
                        if len(gc_imgs_x)==0:
                            gc_img = np.zeros(rgb_img.shape, np.uint8)
                            gc_mask = np.zeros(rgb_img.shape, np.uint8)
                            cv2.imwrite('%s/mask_grabcut_x/%06d_10.png' % (out_dir, test_id),gc_mask)
                        elif len(gc_imgs_x)==1:
                            gc_img = gc_imgs_x[0]
                            gc_mask = gc_masks_x[0]
                            gc_mask_inter = np.interp(gc_mask, (0, 1),(0,255))
                            cv2.imwrite('%s/mask_grabcut_x/%06d_10.png' % (out_dir, test_id),gc_mask_inter)
                        else:
                            gc_img = gc_imgs_x[0]
                            gc_mask = gc_masks_x[0]
                            for img_count in range(1,len(gc_imgs_x)):
                                gc_img = cv2.bitwise_or(gc_img, gc_imgs_x[img_count])
                                gc_mask = cv2.bitwise_or(gc_mask, gc_masks_x[img_count])
                                gc_mask_inter = np.interp(gc_mask, (0, 1),(0,255))
                            cv2.imwrite('%s/mask_grabcut_x/%06d_10.png' % (out_dir, test_id),gc_mask_inter)

                        # ***************************** END: GRABCUT SEGMENTATION ***************************** 

                        # ***************************** START: FUSION MODEL - YOLO + GRABCUT FOR SEGMENTATION OPTIMIZING ***************************** 
                        # -- Set paths --
                        weights_path = '$PATH_TO_YOLO_WEIGHTS/yolo_weights/yolov7-seg.pt'      #YOUR CODE: Give absolute path
                        yolo_inf_path = '$PATH_TO_YOLO_SEGMENTATION_SCRIPT/segment/predict.py' #YOUR CODE: Give absolute path
                        roi_dir = '%s/temp_RGB_roi/%06d_10' % (out_dir, test_id)
                        yolo_roi_dir = '%s/temp_YOLO_roi/%06d_10' % (out_dir, test_id)
                        yolo_pred_dir = '$PATH_TO_YOLO_RUNS/runs/predict-seg'                  #YOUR CODE: Give absolute path
                        python_path = sys.executable

                        # -- Create a subdirectory in the Yolo ROIs with current id name (for buffer, for yolo out) --
                        os.makedirs(roi_dir, exist_ok=True)
                        os.makedirs(yolo_roi_dir, exist_ok=True)

                        # -- Read and make a copy of the grabcut image --
                        current_gc_z = cv2.imread('%s/mask_grabcut_z/%06d_10.png' % (out_dir, test_id), cv2.IMREAD_UNCHANGED)
                        current_gc_x = cv2.imread('%s/mask_grabcut_x/%06d_10.png' % (out_dir, test_id), cv2.IMREAD_UNCHANGED)

                        gc_optim_masks_z, gc_optim_masks_x, yolo_masks_t, fusion_masks  = [], [], [], []
                        # -- Loop for all VALID regions --
                        labels = csv.reader(open(labels_file), delimiter=" ")
                        for count, label in enumerate(labels):
                            gc_optim_mask_z = np.zeros(selected_thresh_z.shape, np.uint8)
                            gc_optim_mask_x = np.zeros(selected_thresh_x.shape, np.uint8)
                            yolo_mask_t     = np.zeros(selected_thresh_z.shape, np.uint8)
                            fusion_mask     = np.zeros(rgb_img.shape[:2],np.uint8)

                            # -- Flags to conveniently check motion states in z,x and t axes --
                            z_found = False
                            x_found = False
                            t_found = False
                            if (count in final_motion_picks): 
                                # -- Flag for track only --
                                non_gc_xz = False
                                if (count not in idx_keep_z) and (count not in idx_keep_x):
                                    non_gc_xz = True

                                # -- Read the rgb ROI and save (to be reread by YOLO) --
                                yolo_roi = rgb_img[ int(float(label[2])):int(float(label[4])) , int(float(label[1])):int(float(label[3]))].astype(np.uint8)
                                
                                # -- Set save ROI name (have to save it because yolo takes in saved image as source and not from memory) --
                                tmp_img = '%06d_10_%s.png'%(test_id, count)  
                                cv2.imwrite('%s/%s' % (roi_dir, tmp_img),yolo_roi)

                                """
                                NOTE:we have included cyclists and pedestrians for the segmentation fusion model. 
                                     However, it is redundant in the current state of the model as we consider only 
                                     vehicles. We hope in the future to include VRUs in our predictions and hence 
                                     make use of including these categories within the segmentation architecture
                                """
                                # -- Check if it is a cyclist using KITTI label --
                                cyclist_flag = False
                                if str(label[0])== "Cyclist:":
                                    cyclist_flag = True

                                # -- Clear YOLO pred directory --
                                for root, dirs, files in os.walk(yolo_pred_dir, topdown=False):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        os.remove(file_path)
                                    for dir in dirs:
                                        dir_path = os.path.join(root, dir)
                                        os.rmdir(dir_path)

                                # -- Get kitti label to check if pedestrian, cyclist or vehicle --
                                kitti_ru = str(label[0])
                                if (kitti_ru == 'Pedestrian:'):
                                    kitti_ru_idx = 0
                                elif (kitti_ru == 'Cyclist:'):
                                    kitti_ru_idx = 1
                                else:
                                    kitti_ru_idx = 2

                                # -- Apply YOLO v7 segmentation (custom params already set in YOLOv7 scripts) for the current ROI --    
                                image_path = roi_dir+'/'+tmp_img  
     
                                subprocess.run([python_path, yolo_inf_path, '--weights', weights_path, '--source', image_path, 
                                '--classes','0','1','2','3','5','6','7', '--log_ds', log_out_txt, '--cyclist_flag', str(cyclist_flag)])
                                
                                # -- Save yolo masked output --
                                seg_in_path = yolo_pred_dir+'/exp/'+tmp_img
                                seg_in = cv2.imread(seg_in_path, cv2.IMREAD_UNCHANGED)
                                cv2.imwrite('%s/%s' % (yolo_roi_dir, tmp_img),seg_in)
                                seg_in = cv2.cvtColor(seg_in, cv2.COLOR_BGR2GRAY)
                                # -- Make sure only 0s and 255s are there --
                                _, seg_in = cv2.threshold(seg_in, 1, 255, cv2.THRESH_BINARY)

                                seg_class_flag = False
                                lbl_path = yolo_pred_dir+'/exp/labels'
                                sum_yolo_mask = 0
                                # -- Transfer and save labels file --                                
                                if len(os.listdir(lbl_path))!=0:
                                    yolo_labels_src = glob.glob(lbl_path+'/*.txt')[0] 
                                    yolo_labels_name = os.path.basename(yolo_labels_src) # filename
                                    yolo_labels_dst = yolo_roi_dir+'/'+yolo_labels_name 
                                    shutil.copy(yolo_labels_src, yolo_labels_dst)
                                    # -- Get sum of yolo mask pixels --
                                    sum_yolo_mask = np.sum(seg_in!=0)
                                    # -- Get Yolo label --
                                    with open(yolo_labels_dst, 'r') as label_file:
                                        instances = label_file.readlines()
                                        if len(instances) == 2:
                                            yolo_ru_idx = 1
                                        else:
                                            yolo_ru = int(instances[0].split()[0])
                                            if yolo_ru == 0:
                                                yolo_ru_idx = 0 #person
                                            elif yolo_ru == 1:
                                                yolo_ru_idx = -1 #cycle with no rider
                                            elif yolo_ru == 3:
                                                yolo_ru_idx = -1 #bike with no rider
                                            else:
                                                yolo_ru_idx = 2 # vehicle
                                    if kitti_ru_idx == yolo_ru_idx:
                                        seg_class_flag = True
                                
                                # -- Mask optim params (NOTE: empirically obtained) -- 
                                sum_thresh_lb = float(self.move_params['seg_fusion']['thresh_fusion'])  # fusion threshold
                                sum_thresh_bb = int(self.move_params['seg_fusion']['thresh_region'])  # region threshold
                                inter_thresh  = float(self.move_params['seg_fusion']['thresh_overlap']) # overlap ratio threshold
                                
                                tmp_img = '%06d_10_%s.png'%(test_id, count)  
                                # -- Execute the segmentation fusion --
                                # z direction
                                z_fusion_roi_dir = '%s/temp_z_fusion_roi/%06d_10' % (out_dir, test_id)
                                os.makedirs(z_fusion_roi_dir, exist_ok=True)
                                if count in idx_keep_z:
                                    z_found = True
                                    gc_optim_masks_z, gc_optim_mask_z = yo_gc_fusion_spatial(z_fusion_roi_dir, current_gc_z, tmp_img, lbl_path, sum_yolo_mask,
                                                                                             sum_thresh_bb, sum_thresh_lb, inter_thresh, label, motion_rois_overlap_dict_z,
                                                                                             count, seg_class_flag, seg_in, gc_optim_mask_z, gc_optim_masks_z)

                                # x direction
                                x_fusion_roi_dir = '%s/temp_x_fusion_roi/%06d_10' % (out_dir, test_id)
                                os.makedirs(x_fusion_roi_dir, exist_ok=True)
                                if count in idx_keep_x:
                                    x_found = True 
                                    gc_optim_masks_x, gc_optim_mask_x = yo_gc_fusion_spatial(x_fusion_roi_dir, current_gc_x, tmp_img, lbl_path, sum_yolo_mask,
                                                                                                      sum_thresh_bb, sum_thresh_lb, inter_thresh, label, motion_rois_overlap_dict_x,
                                                                                                      count, seg_class_flag, seg_in, gc_optim_mask_x, gc_optim_masks_x)
                                
                                # temporal
                                t_fusion_roi_dir = '%s/temp_t_fusion_roi/%06d_10' % (out_dir, test_id)
                                os.makedirs(t_fusion_roi_dir, exist_ok=True)
                                if (non_gc_xz) and (count not in out_of_lidar_bb_ids): # no region in z and x, but exists in t
                                    t_found = True
                                    tmp_rows = int(int(float(label[4])) - int(float(label[2])))
                                    tmp_cols = int(int(float(label[3])) - int(float(label[1])))
                                    gc_roi_t = np.zeros((tmp_rows,tmp_cols), np.uint8) #current_gc_z[ int(float(label[2])):int(float(label[4])) , int(float(label[1])):int(float(label[3]))].astype(np.uint8)
                                    # -- Check if ROI has segmentations by checking labels folder -- 
                                    if (len(os.listdir(lbl_path))!=0):
                                        gc_roi_t[seg_in==0]=0
                                        gc_roi_t[seg_in>0]=255

                                        # -- Set new/optimized ROI --
                                        yolo_mask_t[ int(float(label[2])):int(float(label[4])) , int(float(label[1])):int(float(label[3])) ] = gc_roi_t.astype(np.uint8)
                                        yolo_masks_t.append(yolo_mask_t)
                                    # -- Save fusion in "t direction" --
                                    cv2.imwrite('%s/%s' % (t_fusion_roi_dir, tmp_img),gc_roi_t.astype(np.uint8))

                                # final fusion (inc. fully-occluded elimination)
                                fusion_roi_dir = '%s/temp_fusion_roi/%06d_10' % (out_dir, test_id)
                                os.makedirs(fusion_roi_dir, exist_ok=True)
                                if not (int(label[5])==2):
                                    seg_coords = [int(float(label[1])), int(float(label[2])), int(float(label[3])), int(float(label[4]))]  
                                    seg_line = ' '.join(map(str, seg_coords)) + '\n'

                                    if t_found: # only t (no z and x)
                                        if z_found or x_found:
                                            logging.warning(f'OVERDEFINED SEGMENTATION COMBINATION')
                                        # read t image 
                                        tmp_txt = '%06d_10_%s.txt'%(test_id, count)  
                                        t_seg = cv2.imread('%s/%s' % (t_fusion_roi_dir, tmp_img), cv2.IMREAD_UNCHANGED)

                                        # assign segmentation to region
                                        fusion_mask[ int(float(label[2])):int(float(label[4])) , int(float(label[1])):int(float(label[3])) ] = t_seg.astype(np.uint8)
                                        fusion_masks.append(fusion_mask)

                                        # save segmentation png and txt
                                        cv2.imwrite('%s/%s' % (fusion_roi_dir, tmp_img),fusion_mask.astype(np.uint8))                       
                                        with open('%s/%s' % (fusion_roi_dir, tmp_txt), 'w') as file:
                                            file.write(seg_line)
                                                
                                    elif z_found and not x_found:
                                        # read z image 
                                        tmp_img = '%06d_10_%s.png'%(test_id, count)  
                                        tmp_txt = '%06d_10_%s.txt'%(test_id, count)  
                                        z_seg = cv2.imread('%s/%s' % (z_fusion_roi_dir, tmp_img), cv2.IMREAD_UNCHANGED)

                                        # assign segmentation to region
                                        fusion_mask[ int(float(label[2])):int(float(label[4])) , int(float(label[1])):int(float(label[3])) ] = z_seg.astype(np.uint8)
                                        fusion_masks.append(fusion_mask)

                                        # save segmentation
                                        cv2.imwrite('%s/%s' % (fusion_roi_dir, tmp_img),fusion_mask.astype(np.uint8))
                                        with open('%s/%s' % (fusion_roi_dir, tmp_txt), 'w') as file:
                                            file.write(seg_line)

                                    elif x_found and not z_found:
                                        # read x image 
                                        tmp_img = '%06d_10_%s.png'%(test_id, count) 
                                        tmp_txt = '%06d_10_%s.txt'%(test_id, count) 
                                        x_seg = cv2.imread('%s/%s' % (x_fusion_roi_dir, tmp_img), cv2.IMREAD_UNCHANGED)

                                        # assign segmentation to region
                                        fusion_mask[ int(float(label[2])):int(float(label[4])) , int(float(label[1])):int(float(label[3])) ] = x_seg.astype(np.uint8)
                                        fusion_masks.append(fusion_mask)

                                        # save segmentation
                                        cv2.imwrite('%s/%s' % (fusion_roi_dir, tmp_img),fusion_mask.astype(np.uint8))
                                        with open('%s/%s' % (fusion_roi_dir, tmp_txt), 'w') as file:
                                            file.write(seg_line)

                                    elif x_found and z_found:
                                        # read z and x images
                                        tmp_img = '%06d_10_%s.png'%(test_id, count)  
                                        tmp_txt = '%06d_10_%s.txt'%(test_id, count)
                                        z_seg = cv2.imread('%s/%s' % (z_fusion_roi_dir, tmp_img), cv2.IMREAD_UNCHANGED)
                                        x_seg = cv2.imread('%s/%s' % (x_fusion_roi_dir, tmp_img), cv2.IMREAD_UNCHANGED)
                                        zx_seg = cv2.bitwise_or(z_seg, x_seg)

                                        # assign segmentation to region
                                        fusion_mask[ int(float(label[2])):int(float(label[4])) , int(float(label[1])):int(float(label[3])) ] = zx_seg.astype(np.uint8)
                                        fusion_masks.append(fusion_mask)

                                        # save segmentation
                                        cv2.imwrite('%s/%s' % (fusion_roi_dir, tmp_img),fusion_mask.astype(np.uint8))
                                        with open('%s/%s' % (fusion_roi_dir, tmp_txt), 'w') as file:
                                            file.write(seg_line)

                        # -- Combine the ROI-Yolo+ROI-Grabcut masks --
                        # -- z direction --
                        if len(gc_optim_masks_z)==0:
                            new_gc_optim_z = current_gc_z.copy() # zero image
                        elif len(gc_optim_masks_z)==1:
                            new_gc_optim_z = gc_optim_masks_z[0].copy() # mask array has only one element 
                        else:
                            new_gc_optim_z = gc_optim_masks_z[0].copy()
                            for count in range(1,len(gc_optim_masks_z)):
                                new_gc_optim_z = cv2.bitwise_or(new_gc_optim_z, gc_optim_masks_z[count])
                        # -- x direction --
                        if len(gc_optim_masks_x)==0:
                            new_gc_optim_x = current_gc_x.copy() # zero image
                        elif len(gc_optim_masks_x)==1:
                            new_gc_optim_x = gc_optim_masks_x[0].copy() # mask array has only one element 
                        else:
                            new_gc_optim_x = gc_optim_masks_x[0].copy()
                            for count in range(1,len(gc_optim_masks_x)):
                                new_gc_optim_x = cv2.bitwise_or(new_gc_optim_x, gc_optim_masks_x[count])
                        # -- temporal --
                        if len(yolo_masks_t)==0:
                            new_yolo_mask_t = np.zeros(selected_thresh_z.shape, np.uint8)                        
                        elif len(yolo_masks_t)==1:
                            new_yolo_mask_t = yolo_masks_t[0].copy() # mask array has only one element 
                        else:
                            new_yolo_mask_t = yolo_masks_t[0].copy()
                            for count in range(1,len(yolo_masks_t)):
                                new_yolo_mask_t = cv2.bitwise_or(new_yolo_mask_t, yolo_masks_t[count])

                        # -- fusion --
                        if len(fusion_masks)==0:
                            new_fusion_mask = np.zeros(selected_thresh_z.shape, np.uint8)                        
                        elif len(fusion_masks)==1:
                            new_fusion_mask = fusion_masks[0].copy() # mask array has only one element 
                        else:
                            new_fusion_mask = fusion_masks[0].copy()
                            for count in range(1,len(fusion_masks)):
                                new_fusion_mask = cv2.bitwise_or(new_fusion_mask, fusion_masks[count])
                                
                        # -- Save new image with optimized ROIs --
                        if len(new_gc_optim_z.shape)==3: #z
                            new_gc_optim_z = cv2.cvtColor(new_gc_optim_z, cv2.COLOR_BGR2GRAY)
                        if len(new_gc_optim_x.shape)==3: #x                      
                            new_gc_optim_x = cv2.cvtColor(new_gc_optim_x, cv2.COLOR_BGR2GRAY)
                        if len(new_yolo_mask_t.shape)==3: #t                        
                            new_yolo_mask_t = cv2.cvtColor(new_yolo_mask_t, cv2.COLOR_BGR2GRAY)
                        if len(new_fusion_mask.shape)==3: #fusion                       
                            new_fusion_mask = cv2.cvtColor(new_fusion_mask, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite('%s/fused_segmentations/%06d_10.png' % (out_dir, test_id),new_fusion_mask)
                        
                        # ***************************** END: FUSION MODEL - YOLO + GRABCUT FOR SEGMENTATION OPTIMIZING ***************************** 

                        # -- Save our results as annotations -- 
                        vehicle_cats = ['Car','Van','Truck','Tram','Bus'] # remove if not a type of vehicle --
                        with open(results_out, 'a') as results_out_file:
                            # loop all idxs and check which is stationary and which is moving
                            for count, label in enumerate(idx_all):

                                current_cat = label[0]+':'
                                if current_cat in vehicle_cats:

                                    # write frame id
                                    results_out_file.write(str(int(test_id*2)) + ' ')

                                    # write values
                                    for value in label:
                                        results_out_file.write(value + ' ')

                                    # write neighbourhood status
                                    if count in out_of_lidar_bb_ids:
                                        ool = 'Outside'
                                    else:
                                        ool = 'Inside'
                                    results_out_file.write(ool + ' ')

                                    # write ego status 
                                    results_out_file.write(ego_state + ' ')

                                    # write motion status
                                    m_status = 's'
                                    if count in final_motion_picks:
                                        m_status = 'm'
                                    results_out_file.write(m_status)

                                    # new line
                                    results_out_file.write('\n')

                        # -- Show predicted instances of moving vehicles --
                        add_mask = True # NOTE: set to false if you do not require the visualization of the masked region 
                        final_bbs = viz_predictions(add_mask, rgb_img, lidar_filter)
                        
                        # -- Label the instances for visualization --
                        opt_final_motion_picks = []
                        for count, label in enumerate(idx_all):
                            if (count in final_motion_picks) and (count not in out_of_lidar_bb_ids):
                                final_bbs = cv2.rectangle(final_bbs, (int(float(label[1])),int(float(label[2]))), (int(float(label[3])),int(float(label[4]))), (0,0,255), 2)
                                opt_final_motion_picks.append(count)
                        logging.info(f'Final moving vehicle ids: {opt_final_motion_picks}')
                        cv2.imwrite('%s/motion_predictions/%06d_10.png' % (out_dir, test_id),final_bbs)

                        # -- Save corresponding scene flows of moving regions --
                        flow_3d_dense_save = np.array(flow_3d_dense.reshape(input_h, input_w, 3))
                        new_fusion_mask = np.array(new_fusion_mask)
                        flow_3d_dense_save[new_fusion_mask==0] = -1
                        np.save('%s/motion_flow_matrices/%06d_10.npy' % (out_dir, test_id), flow_3d_dense_save)

            c = c + 1

        # -- Close the file handler to flush the content to the file --
        file_handler.close()

        # -- Destroy the temp folders --
        to_destroy = ['disp_0','disp_c','temp_RGB_roi','temp_YOLO_roi', 'temp_gc_x_roi','temp_gc_z_roi',
                      'temp_x_fusion_roi', 'temp_z_fusion_roi', 'temp_t_fusion_roi','temp_fusion_roi',
                      'mask_grabcut_x','mask_grabcut_z']
        for directory in to_destroy:
            dir_path = os.path.join(f'{out_dir}',directory)
            shutil.rmtree(dir_path)
               
if __name__ == '__main__':

    # -- Parse input arguments --
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True,
                        help='Path to weights')
    parser.add_argument('--ds', required=True,
                        help='Dataset ID')
    parser.add_argument('--refine', action='store_true',
                        help='Rigid refinement for background areas')
    args = parser.parse_args()

    # -- Get config --
    import ruamel.yaml
    with open('conf/test/kitti.yaml', 'r') as file:
        config = ruamel.yaml.safe_load(file)
    config['testset']['root_dir'] = '$PATH_TO_DATASETS_DIRECTORY/datasets/kitti_scene_flow_{}'.format(args.ds) #YOUR CODE: change path to your dataset location (download from provided data links)
    with open('conf/test/kitti.yaml', 'w') as file:
        ruamel.yaml.dump(config, file)

    # -- Load config --
    with open('conf/test/kitti.yaml', encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))
        cfgs.ckpt.path = args.weights
        cfgs.out_dir = 'move_results_ds_{}'.format(args.ds)
        cfgs.refine = args.refine
        cfgs.move_params = 'conf/test/move.yaml'

    # -- Initialize logging info --
    utils.init_logging()

    # -- Set device --
    if torch.cuda.device_count() == 0:
        device = torch.device('cpu')
        logging.info('No CUDA device detected, using CPU for evaluation')
    else:
        device = torch.device('cuda:0')
        logging.info('Using GPU: %s' % torch.cuda.get_device_name(device))
        cudnn.benchmark = True

    # -- Define evaluator object from class --
    evaluator = Evaluator(device, cfgs)
    evaluator.run()
