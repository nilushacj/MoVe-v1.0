## Description: MoVe (Moving Vehicles)
This repository contains the code and supporting materials for MoVe (Motion of Vehicles), a model used for accurately **identifying moving vehicles near an ego-vehicle in various driving scenarios. MoVe integrates scene-flow analysis with temporal tracking to enhance motion classification and segmentation using sensor-fused data. MoVe highlights the potential of combining spatial and temporal analyses for motion prediction.** Its insights contribute to advancing autonomous driving applications, supporting the development of future machine vision models—such as those based on object-centric learning—to enhance perception, intent estimation, control strategies, and safety. Specific details of the model architecture will be made **open source after publication**.

![Video Example](media/sample_result.gif)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/nilushacj/MoVe-v1.0.git
cd MoVe-v1.0
```

2. Create and activate required environment:
```bash
module load mamba  # Only if required in your environment
mamba env create -f environment.yml
source activate moviflow
```

3. Download required data
The preprocessed data (adapted from the original source - KITTI Vision Benchmark Suite [1]) can be downloaded here (with folder name **datasets**): https://drive.google.com/drive/folders/1saPpuKvZ3Ism2Qvf-0ur6L2zDIwZxpt2?usp=sharing, along with calibration data: https://drive.google.com/drive/folders/163qsYUqGWXu06X2lp95JX_uJdm211_hJ?usp=sharing

4. Update paths in the **move_predict.py** and the **conf/kitti.yaml** scripts, as well as the dataset quantity in **kitti.py**. All locations requiring updates have been marked in the script with the prefix **YOUR CODE**

5. Run the model with the **predict_and_segment.sh** bash file
We have structured and provided this as a job submission script (executed on Aalto Univeristy's high-performance computer cluster known as **Triton** [2]). Update lines 2-18, depending on your local execution environment.

    Line 21 contains python command for executive the **move_predict.py** script. Update the value for the **-- ds** argument with one of the following evaluation dataset ids:
    - 0002
    - 0006
    - 0007
    - 0008
    - 0010
    - 0013
    - 0014
    - 0018

    After updating the file, we submit the job with:
```bash
sbatch predict_and_segment.sh
```

6. Checking logs
You could monitor the output of your job by viewing the log file, which will show the execution details and any error messages. To view the log file in real-time:
```bash
tail -f exec_logs.out
```

7. Outputs
Upon execution, the following outputs would be generated: 
- a directory of the results (visualization) with the naming format **move_results_ds_xxxx** containing:
    - grids depicting all located vehicles per frame
    - labelled instances corresponding to predictions of moving vehicles
    - predicted instance segmentations of moving vehicles
    - corresponding scene-flows of pixels associated with moving vehicles (-1 values for the stationary regions)  
- .txt file of all predictions (`results_annots/our_results_xxxx.txt`), in which each value is space-separated.  
Each row corresponds to a single detection represented with 18 columns (left to right):

| Values | Name        | Description                                                                                           |
|--------|-------------|-------------------------------------------------------------------------------------------------------|
| 1      | frame       | Frame ID (divide by two to get the corresponding image result, e.g., 4 = 000002_10.png).              |
| 1      | type        | Describes the type of vehicle: 'Car', 'Van', 'Truck' 'Tram', 'Tram'.                                         |
| 4      | bbox        | 2D bounding box of the object in the image: contains left, top, right, and bottom pixel coordinates.  |
| 1      | occlusion   | Integer indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown. |
| 1      | truncation  | Integer indicating how much the object has left image boundaries: 0 = fully visible, 1 = partly truncated, 2 = largely truncated. |
| 1      | track       | Unique track ID for the object.                                                                      |
| 3      | dimensions  | 3D object dimensions: height, width, and length (in meters).                                         |
| 3      | location    | 3D object location x, y, z in camera coordinates (in meters).                                        |
| 1      | bounds      | 'Inside' denotes within the neighbourhood mask, 'Outside' denotes outside the neighbourhood mask.    |
| 1      | status      | Driving scenario, which is one of the following (refer to the paper for details): 'Regular', 'Turning', or 'Stopped'. |
| 1      | prediction  | Motion prediction result (from MoVe): 's' = not moving, 'm' = moving.                                |


## Acknowledgement
This work uses data from the the KITTI Vision Benchmark Suite (http://www.cvlibs.net/datasets/kitti/) [1,2]. which is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License. We gratefully acknowledge the authors of KITTI for providing this dataset.


## References 
[1] Paul Voigtlaender, Michael Krause, Aljosa Osep, Jonathon Luiten, Berin Balachandar Gnana Sekar, Andreas Geiger, & Bastian Leibe (2019). MOTS: Multi-Object Tracking and Segmentation. In Conference on Computer Vision and Pattern Recognition (CVPR).

[2] Andreas Geiger, Philip Lenz, & Raquel Urtasun (2012). Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. In Conference on Computer Vision and Pattern Recognition (CVPR).

## License
This repository contains multiple components with distinct licenses:

1. **Code:**  
   The source code in this repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

2. **Derived Data and Results:**  
   The derived and preprocessed data (data links provided above in step 3), as well as the results derived from the KITTI dataset are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License (CC BY-NC-SA 3.0).  
   - **Non-commercial use only:** This repository and its contents must not be used for commercial purposes.
   - **Attribution:** If you use or modify the data/results in this repository, you must credit the KITTI dataset as follows:  
     > This work uses data from the KITTI Vision Benchmark Suite ([http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)), licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License.  
   - **Share-alike:** Any modifications or derived data/results must be shared under the same license.

For details about the KITTI dataset license, refer to: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/).






