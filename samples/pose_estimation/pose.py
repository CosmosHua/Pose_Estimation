import os
import cv2
import numpy as np
import paz.processors as pr
from paz.core import ops
import matplotlib.pyplot as plt
from paz.core import Pose6D
from paz.core.ops import Camera
import trimesh


def get_neighbors(image, row, col, window=1):
    neighbor = image[row - window : row + window + 1, col - window : col + window + 1]
    color_values = np.reshape(neighbor, (9, 3))
    return color_values

def compute_poses(mask, true_class_id, class_name, size=[320, 320], dimension=[.1, .1]):
    directory = '/home/incendio/Documents/Thesis/YCBVideo_detector/color_meshes'
    for name in os.listdir(directory):
        class_id = name.split('_')[0]
        if int(class_id) == true_class_id:
            mesh_path = os.path.join(directory, name)
            mesh = trimesh.load(mesh_path)
            vertex_colors = mesh.visual.vertex_colors[:, :3]
    focal_length = size[1]
    camera_center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, camera_center[0]],
                            [0, focal_length, camera_center[1]],
                            [0, 0, 1]], dtype='double')
    camera = Camera(0)
    camera.intrinsics = camera_matrix
    camera.distortion = np.zeros((4, 1))

    (success, rotation, translation) = ops.solve_PNP(points3d, image2d, camera, ops.UPNP)
    pose6D = Pose6D.from_rotation_vector(rotation, translation, class_name)

    dimensions = {class_name: dimension}
    pose = {'pose6D': pose6D, 'image': mask}
    draw = pr.DrawBoxes3D(camera, dimensions)
    poses, projected_points = draw(pose)
    return poses, projected_points
    # model3d = np.array(model3d).astype(np.float32) #(N, 3)
    # points3d, matched_indices = [], []
    # for index in range(len(rows)):
    #     row, col = rows[index], cols[index]
    #     neighbor = get_neighbors(mask, row, col)
    #     x_index = np.where(vertex_colors.all() == neighbor)[0]
    #     if len(x_index) == 1:
    #         points3d.append(mesh.vertices[x_index[0], :])
    #         matched_indices.append([col, row])
    # matched_indices = np.array(matched_indices).astype(np.float32)
    # for i in range(mode3d.shape[0]):
    #     x_index = np.where(vertex_colors == model3d[i, :])[0]
    #     mid_index = int(len(x_index) / 2)
    #     points3d.append(mesh.vertices[x_index[mid_index], :])
