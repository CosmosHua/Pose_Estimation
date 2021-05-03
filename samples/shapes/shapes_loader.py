import numpy as np
import cv2
from mrcnn import utils


class Shapes(utils.Dataset):
    """Generates the shapes synthetic dataset
    """

    def load_shapes(self, count, height, width):
        self.add_class('shapes', 1, 'square')
        self.add_class('shapes', 2, 'circle')
        self.add_class('shapes', 3, 'triangle')

        # Add images
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image('shapes', image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'shapes':
            return info['shapes']
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion,
                                       np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        X, Y, size = dims
        if shape == 'square':
            cv2.rectangle(image, (X - size, Y - size), (X + size, Y + size),
                          color, -1)
        elif shape == 'circle':
            cv2.circle(image, (X, Y), size, color, -1)
        elif shape == 'triangle':
            points = np.array([[(X, Y - size),
                                (X - size/np.sin(np.radians(60)), Y + size),
                                (X + size/np.sin(np.radians(60)), Y + size),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        shape = np.random.choice(["square", "circle", "triangle"])
        color = tuple([np.random.randint(0, 255) for _ in range(3)])
        buffer = 20
        Y = np.random.randint(buffer, height - buffer - 1)
        X = np.random.randint(buffer, width - buffer - 1)
        size = np.random.randint(buffer, height//4)
        return shape, color, (X, Y, size)

    def random_image(self, height, width):
        bg_color = np.array([np.random.randint(0, 255) for _ in range(3)])
        shapes, boxes = [], []
        N = np.random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            X, Y, size = dims
            boxes.append([Y - size, X - size, Y + size, X + size])
        # Apply non-max suppression wit 0.3 threshold to avoid
        keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes
