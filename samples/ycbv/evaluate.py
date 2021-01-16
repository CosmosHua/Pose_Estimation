
import os
import sys
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from ycbv_loader import YCBVDataset

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
data_path = '/gluster/home/sdevaramani/Thesis/10k_data'


class YCBVConfig(Config):
    # Give the configuration a recognizable name
    NAME = 'ycb'
    NUM_CLASSES = 1 + 21  # background + 3 shapes
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # TRAIN_ROIS_PER_IMAGE = 32
config = YCBVConfig()

class InferenceConfig(YCBVConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

dataset_val = YCBVDataset(data_path, split='test')
dataset_val.load_ycbv()
dataset_val.prepare()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

weights_path = '/gluster/home/sdevaramani/Thesis/refactor/versions/mse_logs/for_10k_data/ycb20210110T1048/mask_rcnn_ycb_0090.h5'
model.load_weights(weights_path, by_name=True)

image_ids = np.random.choice(dataset_val.image_ids, 500)
#image_ids = dataset_val.image_ids
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, _, class_id, boxes, r_mask, g_mask, b_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    #molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP over range
    ap = utils.compute_ap_range(boxes, class_id, r_mask, g_mask, b_mask, r['rois'], 
                                r['class_ids'], r['scores'], r['r_masks'], r['g_masks'], r['b_masks'], verbose=0)

    #AP, precisions, recalls, overlaps =\
    #    utils.compute_ap(boxes, class_id, r_mask, g_mask, b_mask,
    #                     r['rois'], r['class_ids'], r['scores'], r['r_masks'], r['g_masks'], r['b_masks'])
    APs.append(ap)

#print("mAP: ", np.mean(APs))
print("Mean AP over {} images: {:.4f}".format(len(APs), np.mean(APs)))
