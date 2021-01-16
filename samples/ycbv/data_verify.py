import os
import sys
import numpy as np
import cv2
import tensorflow as tf
import json
from json import JSONEncoder

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../..")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from ycbv_loader import YCBVDataset


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class YCBVConfig(Config):
    NAME = 'ycb'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 21
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    TRAIN_ROIS_PER_IMAGE = 100
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 5

config = YCBVConfig()

class_names = ['background', '002_master_chef_can', '003_cracker_box',
               '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
               '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
               '010_potted_meat_can', '011_banana', '019_pitcher_base',
               '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill',
               '036_wood_block', '037_scissors', '040_large_marker', '051_large_clamp',                                                                                                              '052_extra_large_clamp', '061_foam_brick']

data_dir = os.path.join(ROOT_DIR, '10k_data')
dataset_train = YCBVDataset(data_dir, split='train')
dataset_train.load_ycbv()
dataset_train.prepare()

image_id = np.random.choice(dataset_train.image_ids)
image, _, class_id, boxes, r_mask, g_mask, b_mask = modellib.load_image_gt(dataset_train, config,
                                                                           image_id, use_mini_mask=False)

r_mask = (r_mask * 255).astype(np.uint8)
g_mask = (g_mask * 255).astype(np.uint8)
b_mask = (b_mask * 255).astype(np.uint8)

masks = []
for i in range(r_mask.shape[-1]):
    mask = np.stack([r_mask[:, :, i], g_mask[:, :, i], b_mask[:, :, i]], axis=2)
    masks.append(mask)

masks = np.stack(masks, axis=2)
masks = np.reshape(masks, (320, 320, 3*7))

result = visualize.display_rgb_instances(image, boxes, masks, class_id, class_names, figsize=(10, 10))

data = {'class_id': class_id, 'boxes':boxes, 'masks': masks}
with open('data_inspection.json', 'w') as fp:
    json.dump(data, fp, cls=NumpyArrayEncoder)
cv2.imwrite('original.png', image)
cv2.imwrite('data_check.png', result)
