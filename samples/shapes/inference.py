import numpy as np
import json
from json import JSONEncoder
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn.model import MaskRCNN
from shapes_loader import ShapesDataset

MODEL_DIR = '/gluster/home/sdevaramani/Thesis'
WEIGHTS_PATH = '/gluster/home/sdevaramani/Thesis/shapes20201104T2321/'\
               'mask_rcnn_shapes_0002.h5'


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class ShapesConfig(Config):
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


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = ShapesConfig()
inference_config = InferenceConfig()

dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

model = MaskRCNN(mode='inference', 
                 config=inference_config,
                 model_dir=MODEL_DIR)

model.load_weights(WEIGHTS_PATH, by_name=True)

image_id = np.random.choice(dataset_val.image_ids)
image, _, _, _, _ = modellib.load_image_gt(dataset_val, inference_config,
                                           image_id, use_mini_mask=False)

results = model.detect([image], verbose=1)
r = results[0]
data = {'rois': r['rois'], 'class_ids': r['class_ids'],
        'scores': r['scores'], 'masks': r['masks']}

with open('results.json', 'w') as fp:
    json.dump(data, fp, cls=NumpyArrayEncoder)
