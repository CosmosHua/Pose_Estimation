import cv2
import os
import trimesh
import numpy as np
from paz.core import Pose6D
from paz.core.ops import Camera
import paz.processors as pr
from paz.core import ops
import matplotlib.pyplot as plt

MESH_DIR = '/home/incendio/Documents/Thesis/YCBVideo_detector/color_meshes'
GREEN = (0, 255, 0)

class PnPSolver():
    """ Implements PnP RANSAC algorithm to compute rotation and 
        translation vector for a given RGB mask of an object
    # Arguments:
        rgb_mask: RGB mask of object
        true_id: Int
        class_name: class name of object. String
        dimension: (width, height) for draw_cube
        size: size of the mask
    """
    def __init__(self, rgb_mask, true_id, class_name, color=GREEN, dimension=[.1, .1], size=(320, 320)):
        self.rgb_mask = rgb_mask
        self.size = size
        self.dimension = dimension
        self.camera = self.compute_camera_matrix()
        self.id = true_id
        self.class_name = class_name
        self.vertex_colors = self.get_vertex_colors()
        self.color = color
        self.world_to_camera = np.array([[ 0.70710678, 0., -0.70710678, 0.01674194],
                                         [-0.40824829, 0.81649658, -0.40824829, -0.01203142],
                                         [ 0.57735027, 0.57735027, 0.57735027, -1.73205081],
                                         [ 0., 0., 0., 1.]])

    def get_vertex_colors(self):
        for name in os.listdir(MESH_DIR):
            class_id = name.split('_')[0]
            if int(class_id) == self.id:
                mesh_path = os.path.join(MESH_DIR, name)
                self.mesh = trimesh.load(mesh_path)
                vertex_colors = self.mesh.visual.vertex_colors[:, :3]
        return vertex_colors

    def solve_PnP(self):
        points3d, image2D = self.get_points()
        assert image2D.shape[0] == points3d.shape[0]
        (_, rotation, translation, inliers) = ops.solve_PNP(points3d, image2D, self.camera, ops.UPNP)
        pose6D = Pose6D.from_rotation_vector(rotation, translation, self.class_name)
        return pose6D

    def visualize_3D_boxes(self, image, pose6D):
        dimensions = {self.class_name: self.dimension}
        pose = {'pose6D': pose6D, 'image': image, 'color': self.color}
        draw = pr.DrawBoxes3D(self.camera, dimensions)
        args, projected_points = draw(pose)
        return args, projected_points


    def get_points(self):
        points3d, image2d = [], []
        rows, cols, channels = np.where(self.rgb_mask > 0)
        for index in range(len(rows)):
            x, y = rows[index], cols[index]
            R, G, B = self.rgb_mask[x, y, :]
            matches = np.unique(np.array(self.get_matches(x, y)))
            if len(matches) == 1:
                image2d.append([y, x])
                vertex = self.mesh.vertices[matches[0], :]
                points3d.append(vertex)
            # x_index = np.where(self.vertex_colors == np.stack([R, G, B]))[0]
            # mid_index = int(len(x_index) / 2)
        # points3d.append(self.mesh.vertices[x_index[mid_index], :])
        image2d = np.array(image2d).astype(np.float32) #(N, 2)
        points3d = np.array(points3d).astype(np.float32) #(N, 3)
        return points3d, image2d

    def get_matches(self, x, y):
        R, G, B = self.rgb_mask[x, y, :]
        r_index = np.where(self.vertex_colors[:, 0] == R)[0]
        g_index = np.where(self.vertex_colors[:, 1] == G)[0]
        b_index = np.where(self.vertex_colors[:, 2] == B)[0]
        matches = [r_index, g_index, b_index]
        intersection = list(set(matches[0]).intersection(*matches))
        return intersection


    def get_model_point(self):
        rows, cols, channels = np.where(self.rgb_mask > 0)
        x, y = int(np.mean(rows)), int(np.mean(cols))
        R, G, B = self.rgb_mask[x, y, 0], self.rgb_mask[x, y, 1], self.rgb_mask[x, y, 2]
        x_index = np.where(self.vertex_colors == np.stack([R, G, B]))[0]
        mid_index = int(len(x_index) / 2)
        return self.mesh.vertices[x_index[mid_index], :]


    def compute_camera_matrix(self):
        focal_length = self.size[1]
        camera_center = (self.size[1] / 2, self.size[0] / 2)
        camera_matrix = np.array([[focal_length, 0, camera_center[0]],
                                  [0, focal_length, camera_center[1]],
                                  [0, 0, 1]], dtype='double')
        camera = Camera(0)
        camera.intrinsics = camera_matrix
        camera.distortion = np.zeros((4, 1))
        return camera

    def draw_axis(self, mask, projected_points, thickness=2):
        rows, cols, channels = np.where(mask > 0)
        x, y = (int(np.mean(rows)), int(np.mean(cols)))
        center = (y, x)
        image = mask.copy()
        R, G, B = (255, 0, 0), (0, 255, 0), (0, 0, 255)
        projected_points = projected_points.astype(np.int32)
        image = cv2.line(image, center, tuple(projected_points[0].ravel()), R, thickness)
        image = cv2.line(image, center, tuple(projected_points[1].ravel()), G, thickness)
        image = cv2.line(image, center, tuple(projected_points[2].ravel()), B, thickness)
        return image

    def get_neighbors(self, image, row, col, window=1):
        neighbor = image[row - window : row + window + 1, col - window : col + window + 1]
        color_values = np.reshape(neighbor, (9, 3))
        return color_values
