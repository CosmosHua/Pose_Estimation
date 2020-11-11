import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import json

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
from json import JSONEncoder
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from ycbv_loader import YCBVDataset
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
data_path = '/gluster/home/sdevaramani/Thesis/randomized_data'

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


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

dataset_val = YCBVDataset(data_path, split='val')
dataset_val.load_ycbv()
dataset_val.prepare()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
#print('.....................', os.walk(MODEL_DIR))

#model_path = model.find_last()

# Load trained weights
#print("Loading weights from ", model_path)

weights_path = '/gluster/home/sdevaramani/Thesis/refactor/logs/ycb20201111T1506/mask_rcnn_ycb_0005.h5'
model.load_weights(weights_path, by_name=True)

#model.load_weights(model_path, by_name=True)
image_id = random.choice(dataset_val.image_ids)

print('image id................', image_id)
original_image, _, gt_class_ids, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                 image_id, use_mini_mask=False)

print(gt_bbox)
gt_data = {'rois': gt_bbox, 'class_ids':gt_class_ids, 'masks': gt_mask}
cv2.imwrite('original_image.png', original_image)
results = model.detect([original_image], verbose=1)
 
r = results[0]
data = {'rois': r['rois'], 'class_ids': r['class_ids'], 'scores': r['scores'], 'masks': r['masks']}

print('results ....', r['rois'])
print(r['class_ids'])
print(r['masks'].shape)
with open('gt.json', 'w') as f:
    json.dump(gt_data, f, cls=NumpyArrayEncoder)

with open('results.json', 'w') as fp:
    json.dump(data, fp, cls=NumpyArrayEncoder)

