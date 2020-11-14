
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log
from ycbv_loader import YCBVDataset
 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
data_path = '/gluster/home/sdevaramani/Thesis/randomized_data'


class YCBVConfig(Config):
    # Give the configuration a recognizable name
    NAME = 'ycb'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 21  # background + 3 shapes
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
config = YCBVConfig()

class InferenceConfig(YCBVConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Validation dataset
dataset_val = YCBVDataset(data_path, split='val')
dataset_val.load_ycbv()
dataset_val.prepare()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

weights_path = '/gluster/home/sdevaramani/Thesis/refactor/logs/ycb20201113T0254/mask_rcnn_ycb_0050.h5'
model.load_weights(weights_path, by_name=True)

image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))
