# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.xml                # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import logging

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils_yolo.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils_yolo.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils_yolo.plots import Annotator, colors, save_one_box
from utils_yolo.segment.general import process_mask, scale_masks
from utils_yolo.segment.plots import plot_masks
from utils_yolo.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.80,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=2,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        log_ds=ROOT / 'log_ds_unspecified.txt',
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, out = model(im, augment=augment, visualize=visualize)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # --- Function to get bounding box dimensions, coordinates and areas from det ---
            def append_dim_coord(in_det):
                # -- Array to store 2D dimensions of ROI --
                dim_2D_array = []
                # -- Array to store 2D coordinates of ROI xyxy--
                dim_2D_coords = []
                # -- Loop rows in the output --
                for idx in range(in_det.shape[0]):
                    # -- Append dimensions (sorted according to original det i.e. confidence) --
                    temp_width = in_det[idx][2] - in_det[idx][0] 
                    temp_height = in_det[idx][3] - in_det[idx][1]
                    dim = (temp_width.cpu().item(), temp_height.cpu().item())
                    dim_2D_array.append(dim) 
                    # -- Append coords (sorted according to original det i.e. confidence) --
                    temp_coords = [in_det[idx][0].cpu().item(), in_det[idx][1].cpu().item(), in_det[idx][2].cpu().item(), in_det[idx][3].cpu().item()]
                    dim_2D_coords.append(temp_coords)
                # -- Get all bb areas (sorted according to original det i.e. confidence) -- 
                areas = [temp_width * temp_height for temp_width, temp_height in dim_2D_array]
                return dim_2D_array, dim_2D_coords, areas

            logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S    ',
                    filename=str(log_ds),
                    filemode='a')
            logger = logging.getLogger()
            # ---- Create a FileHandler to redirect logging output to a file ----
            file_handler = logging.FileHandler(str(log_ds))
            logger.addHandler(file_handler)

            if len(det):
                #print(f'SHAPE OF BOUNDING BOXESS:{det[:, :4].shape}')
                #print(f'SHAPE OF MASKSSSS:{det[:, 6:].shape}')
                logging.info(f'Number of detected instances: {det.shape[0]}')
                if len(det) == 1:
                    temp = det[:, 5]
                    temp = temp.cpu().numpy()
                    temp = temp.astype(int)[0]
                    logging.info(f'Class of single instance: {temp}')
                # ---- Get the 2 bounding boxes with the highest areas if there are at least two ----
                if len(det) > 1:
                    dim_2D, coords_2D, det_areas = append_dim_coord(det)
                    # -- Sort area indexes in descending order --
                    idx_area_sorted = sorted(range(len(dim_2D)), key=lambda i: det_areas[i], reverse=True)
                    logging.info('Classes of multiple instances:')
                    # -- Log detected classes in descending order of area --
                    for desc_count in range(len(idx_area_sorted)):
                        temp_class = det[idx_area_sorted[desc_count], 5].cpu().item()
                        temp_class = int(temp_class) #.astype(int)
                        temp_area  = det_areas[idx_area_sorted[desc_count]]
                        temp_conf  = det[idx_area_sorted[desc_count], 4].cpu().item()
                        temp_conf  = float(temp_conf) #.astype(float)
                        logging.info('    %s (area: %s pixels, confidence: %.2f)'%(temp_class, int(temp_area), temp_conf))
                    if len(det) > 2:
                        # -- Get row indexes of top two --
                        rmd_idx_area_sorted = idx_area_sorted[:2]
                        # -- Remove all rows apart from the top 2 (based on area) --
                        pass_idx = torch.tensor(rmd_idx_area_sorted, device=device)
                        filter_mask = torch.zeros(det.shape[0], dtype=torch.bool, device=device)
                        filter_mask[pass_idx] = True
                        det_new = det[filter_mask]
                        det = det_new.clone() # -- new det made with new indexes --
                    #logging.info(f'ELEMENT 6: {det[:, :6]}')

                    #NOTE: At this point, only two rows remain in det!!
                    # ---- Put remaining class indexes to an array (int & unsorted) ----
                    remaining_classes = det[:, 5]
                    remaining_classes = remaining_classes.cpu().numpy()
                    remaining_classes = remaining_classes.astype(int)
                    #logging.info(f'Filtered classes: {remaining_classes}')

                    # ---- Get areas again (since idxs changed to just 2 and we didn't sort det) ----
                    bi_dim_2D, bi_coords_2D, bi_det_areas = append_dim_coord(det)
                    # ---- Sort the two area indexes in descending order (so output is either [0,1] or [1,0]) ----
                    bi_idx_area_sorted = sorted(range(len(bi_dim_2D)), key=lambda i: bi_det_areas[i], reverse=True)
                    logging.info('Remaining classes:')
                    # -- Log remaining classes in descending order of area --
                    for desc_count in range(len(bi_idx_area_sorted)):
                        temp_class = det[bi_idx_area_sorted[desc_count], 5].cpu().item()
                        temp_class = int(temp_class)
                        temp_area  = bi_det_areas[bi_idx_area_sorted[desc_count]]
                        temp_conf  = det[bi_idx_area_sorted[desc_count], 4].cpu().item()
                        temp_conf  = float(temp_conf)
                        logging.info('    %s (area: %s pixels, confidence: %.2f)'%(temp_class, int(temp_area), temp_conf))

                    # ---- If the 2 bounding boxes remaining are bicycle/bike and a person ----
                    if set(remaining_classes) == {0, 1} or set(remaining_classes) == {0, 3}:
                        logging.info('Possible cyclist found!')
                        cyclist_thresh = 0.20
                        # -- Get coordinates of the two bbs --
                        roi_1 = bi_coords_2D[0]
                        roi_2 = bi_coords_2D[1]
                        x_0 = np.max([roi_1[0], roi_2[0]])
                        x_1 = np.min([roi_1[2], roi_2[2]])
                        y_0 = np.max([roi_1[1], roi_2[1]])
                        y_1 = np.min([roi_1[3], roi_2[3]])
                        # --- Get the intersection area/ratio of the two boxes ---
                        if (x_1 < x_0) or (y_1 < y_0):
                            inter_ratio  = 0.0
                        else:
                            inter_area  = (x_1 - x_0) * (y_1 - y_0)
                            roi_1_area  = (roi_1[2] - roi_1[0]) * (roi_1[3] - roi_1[1]) 
                            roi_2_area  = (roi_2[2] - roi_2[0]) * (roi_2[3] - roi_2[1]) 
                            tot_area    = (roi_1_area + roi_2_area) - inter_area
                            inter_ratio = inter_area / tot_area
                        logging.info(f'Intersection ratio: {inter_ratio}')

                        # --- Remove the second row from detections if it is not a cyclist ---
                        if inter_ratio < cyclist_thresh:
                            idx_to_remove = bi_idx_area_sorted[-1]
                            #logging.info(f'INDEX TO REMOVE: {idx_to_remove}')

                            # -- Create a mask to exclude the row to remove --
                            filter_mask = torch.ones(det.shape[0], dtype=torch.bool, device=device)
                            filter_mask[idx_to_remove] = False
                            # -- Apply the mask to the tensor to remove the row --
                            det_not_cyc = det[filter_mask]
                            # -- Overwrite original output --
                            det = det_not_cyc.clone()
                    # ---- If the 2 bounding boxes remaining are NOT a combination of bicycle/bike and a person ----
                    else:
                        idx_to_remove = bi_idx_area_sorted[-1]
                        # -- Create a mask to exclude the row to remove --
                        filter_mask = torch.ones(det.shape[0], dtype=torch.bool, device=device)
                        filter_mask[idx_to_remove] = False
                        # -- Apply the mask to the tensor to remove the row --
                        det_not_cyc = det[filter_mask]
                        # -- Overwrite original output --
                        det = det_not_cyc.clone()
                logging.info(f'Number of final instances: {det.shape[0]}') # always 1, but 2 for cyclist

                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                # Mask plotting ----------------------------------------------------------------------------------------

                # Write results
                for *xyxy, conf, cls in reversed(det[:, :6]):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            else:
                logging.info('Number of detected instances: 0')

            file_handler.close()
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.10, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=10, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--log_ds', type=str, help='log output file')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
