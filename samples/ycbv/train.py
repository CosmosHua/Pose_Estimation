import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import tensorflow as tf
ROOT_DIR = os.path.abspath("../../")

from mrcnn.model import data_generator
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from ycbv_loader import YCBVDataset

# Directory to save logs and trained model
#MODEL_DIR = os.path.join(ROOT_DIR + '/Thesis', "logs")
MODEL_DIR = '/gluster/home/sdevaramani/Thesis/refactor/versions/binary_cross_entropy_logs'
COCO_MODEL_PATH = '/gluster/home/sdevaramani/Thesis/weights/mask_rcnn_coco.h5'
data_path = '/gluster/home/sdevaramani/Thesis/50_images'


class YCBVConfig(Config):
    # Give the configuration a recognizable name
    NAME = 'ycb'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 21  # background + 3 shapes
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    #TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 500

config = YCBVConfig()

# Training dataset
dataset_train = YCBVDataset(data_path, split='train')
dataset_train.load_ycbv()
dataset_train.prepare()

# Validation dataset
dataset_val = YCBVDataset(data_path, split='val')
dataset_val.load_ycbv()
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=3, 
            layers='heads')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=5,
            layers='4+')

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=10, 
            layers="all")
