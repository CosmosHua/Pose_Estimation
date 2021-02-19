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
  - Create an empty directory for saving the domain randomized data as following format
    +-- Data
    | +-- Train
      | +-- rgb
      | +-- mask
      | +-- images
    | +-- class_info.json
   Similar to Train/ directory create test and val inside /Data. Then copy the following into `class_info.json`
   ```
   {"00": "background", "01": "002_master_chef_can", "02": "003_cracker_box", "03": "004_sugar_box", 
   "04": "005_tomato_soup_can", "05": "006_mustard_bottle", "06": "007_tuna_fish_can", 
   "07": "008_pudding_box", "08": "009_gelatin_box", "09": "010_potted_meat_can", "10": 
   "011_banana", "11": "019_pitcher_base", "12": "021_bleach_cleanser", "13": "024_bowl", 
   "14": "025_mug", "15": "035_power_drill", "16": "036_wood_block", "17": "037_scissors", 
   "18": "040_large_marker", "19": "051_large_clamp", "20": "052_extra_large_clamp", "21": "061_foam_brick"}
   ```
  - Run renderer.py as: 
  ```
  python3 renderer.py -mp 'path/to/object/models' -rp 'path/to/save/rendered/data' 
  -s 'train/test/val' --num_objects 7 --sample_size data_size
  ```
  - Now add background and augment data
    ```
    python3 data_augmentation.py -dp 'path/to/saved/rendered/data' -bp 'path/to/background/images' 
    -s 'train/val/test
    ```
  - Generate ground-truth boxes for the augmented data
  
    `python3 generate_groundtruth.py -dp 'path/to/saved/augmented/data' -s 'train/test/val'`
    
 * Step 3: Inspect data as in `inspect_data.ipynb` to verfiy the generated data
 * Step 4: Before training the model, download `mask_rcnn_coco.h5` from this [link](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) as we initialize weights with Mask RCNN trained for COCO.
 * Step 5: Change the MODEL_DIR (path to save trained model), data_path (path to dataset) and COCO_MODEL_DIR (path to saved COCO weights) in `samples/ycbv/train` and run the script to start the training.
 * Step 6: After training the model, run `samples/ycbv/inference.py` to run model inference and the results will be saved in your directory as `results.json`
 * Step 7: To visualize the resulting bounding boxes and masks, run `inspect_inference.ipynb`.



    `python3 samples/ycbv/train.py`
