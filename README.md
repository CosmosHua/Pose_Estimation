# Efficient architectures for object Pose estimation

This repository consists of code corresponding the thesis titled - 'Efficient architectures for object pose estimation'


Implementation Details:

* The repository consists of implementation of reformed Mask R-CNN model that performs object detection and instance segmentation via RGB masks.
* The source code adapted is [Matterport Mask RCNN](https://github.com/matterport/Mask_RCNN)
* The directory `mrcnn` includes the implementation of complete reformed Mask R-CNN
* The directory  `paz` comprises of modules from [Perception for Autonomous Systems (PAZ)](https://github.com/oarriaga/paz) library.
* `samples/ycbv`: Includes training and inference scripts for reformed model
* `samples/pose_estimation`: Includes code for pose estimation and evaluation
* `samples/domain_randomization`: Includes code for generating domain randomized data for YCB-Video dataset


Setting up:

* Step 1: To install the required dependencies and build the environment run the following command:

`pip install . --user` 

* Step 2: Generation of domain randomized data
  - Download the model files (.obj) for YCB-Video dataset from [BOP](http://ptak.felk.cvut.cz/6DB/public/bop_datasets/ycbv_models.zip) 
  - Run renderer.py as: `python3 renderer.py -mp 'path/to/object/models' -rp 'path/to/save/rendered/data' -s 'train/test/val' --num_objects 7 --sample_size data_size`
