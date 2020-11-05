import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class YCBVDataset(utils.Dataset):
    """ Loads YCB-Video dataset for Mask RCNN training
    """
    def __init__(self, data_dir, split=None, name='ycb'):
        super(YCBVDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.name = name
        self.mask_files = self.set_all_mask_files()

    def set_all_mask_files(self):
        mask_files = {}
        masks_dir = os.path.join(self.data_dir, self.split + '/mask')
        for file in os.listdir(masks_dir):
            mask_file = masks_dir + '/' + file
            IMID, class_id = file.split('_')
            class_id, _ = class_id.split('.')
            mask_files.setdefault(IMID, {}).update({class_id: mask_file})
        return mask_files

    def load_ycbv(self):
        class_names = self.get_class_names()
        for class_id in range(1, len(class_names)):
            self.add_class('ycb', class_id, class_names[class_id])
        assert self.split in ['train', 'val']
        images = {}
        images_dir = os.path.join(self.data_dir, self.split + '/rgb')
        for filename in os.listdir(images_dir):
            IMID, _ = filename.split('.')
            images[IMID] = images_dir + '/' + filename
        images = dict(sorted(images.items(), key=lambda x: x[1]))
        for image_id in list(images.keys()):
            self.add_image('ycb', image_id, path=images[image_id])
        self.image_info = sorted(self.image_info, key=lambda k: int(k['id']))

    def get_class_names(self):
        class_names = ['background', '002_master_chef_can', '003_cracker_box',
                       '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                       '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
                       '010_potted_meat_can', '011_banana', '019_pitcher_base',
                       '021_bleach_cleanser', '024_bowl',
                       '025_mug', '035_power_drill',
                       '036_wood_block', '037_scissors',
                       '040_large_marker', '051_large_clamp',
                       '052_extra_large_clamp', '061_foam_brick']
        return class_names

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'ycb':
            return info['ycb']
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        image_masks = self.mask_files[str(image_id)]
        instance_masks = []
        class_ids = []
        for class_id in list(image_masks.keys()):
            bgr_mask = cv2.imread(image_masks[class_id])
            binary_mask = np.clip(bgr_mask, 0, 1)
            instance_masks.append(binary_mask[:, :, 0])
            class_ids.append(class_id)
        class_ids = np.array(class_ids, dtype=np.int32)
        mask = np.stack(instance_masks, axis=2).astype(np.bool)
        return mask, class_ids

    def load_rgb_mask(self, image_id):
        image_masks = self.mask_files[str(image_id)]
        instance_masks = []
        class_ids = []
        for class_id in list(image_masks.keys()):
            bgr_mask = cv2.imread(image_masks[class_id])
            rgb_mask = cv2.cvtColor(bgr_mask, cv2.COLOR_BGR2RGB)
            instance_masks.append(rgb_mask)
            class_ids.append(class_id)
        class_ids = np.array(class_ids, dtype=np.int32)
        mask = np.stack(instance_masks, axis=2)
        mask = np.reshape(mask, (640, 640, 3 * len(class_ids)))
        return mask, class_ids
