## Description

This repo is for creating a dataloader for a custom tensorflow dataset for Waymo Perception data. The aim is to make the dataset compatible for being used as input to a JAX-based pipeline motivated by the SAVi++ object centric learning model for autonomous driving

## Data download


## Directory Structure

- **README.md** 
- **brno_encoder_parallel_concurrent.py**
- **brno_winter_encoder_parallel_concurrent.py**
- **custom_data_loader_brno.py**
- **custom_data_loader_waymo.py**
- **open_test_funcs.py**
- **open_test_funcs_brno.py**
- **test_data_reader_movi.py** 
- **test_data_reader_waymo.py** 
- **movi_e/**: 
  - **128x128/**: 
    - **1.0.0/**: 
      - **dataset_info.json** 
      - **features.json**
      - **instances-category.labels.txt**
      - ***movi tfrecord files***
      
- **waymo/**: 
  - **128x192/**: 
    - **1.0.0/**: 
      - **dataset_info.json** 
      - **features.json**
      - **waymo-train.tfrecord-00001-of-00002**
      - **waymo-train.tfrecord-00002-of-00002**

- **brno/**: 
  - **128x192/**: 
    - **1.0.0/**: 
      - **dataset_info.json** 
      - **features.json**
      - **brno-train.tfrecord-00001-of-00002**
      - **brno-train.tfrecord-00002-of-00002**

- **brno-normal/**: 
  - **128x192/**: 
    - **1.0.0/**: 
      - **dataset_info.json** 
      - **features.json**
      - **brno-train.tfrecord-00001-of-00002**
      - **brno-train.tfrecord-00002-of-00002**

## Support material


# Pre-requisites
Savi++ virtual environment must be installed: https://github.com/google-research/slot-attention-video/

# Execution for Waymo
