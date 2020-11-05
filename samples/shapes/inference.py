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
from shapes_loader import ShapesDataset
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class ShapesConfig(Config):
# Give the configuration a recognizable name
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    NUM_CLASSES = 1 + 3  # background + 3 shapes
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5
config = ShapesConfig()

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
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

weights_path = '/gluster/home/sdevaramani/Thesis/shapes20201104T2321/mask_rcnn_shapes_0002.h5'
model.load_weights(weights_path, by_name=True)

#model.load_weights(model_path, by_name=True)
image_id = random.choice(dataset_val.image_ids)

print('image id................', image_id)
original_image, _, _, _, _ = modellib.load_image_gt(dataset_val, inference_config,
                                                 image_id, use_mini_mask=False)
 
cv2.imwrite('original_image.png', original_image)
results = model.detect([original_image], verbose=1)
 
r = results[0]
data = {'rois': r['rois'], 'class_ids': r['class_ids'], 'scores': r['scores'], 'masks': r['masks']}

with open('results.json', 'w') as fp:
    json.dump(data, fp, cls=NumpyArrayEncoder)

