import threading
import traceback
from torch.multiprocessing import Process, Queue, Pool
import numpy as np

import os
import torch


def get_multipose_test_input(data, render, yaw_poses, pitch_poses):
    real_image = data['image']
    num_poses = len(yaw_poses) + len(pitch_poses)
    rotated_meshs = []
    rotated_landmarks_list = []
    original_angles_list = []
    rotated_landmarks_list_106 = []
    paths = []
    real_images = []
    pose_list = []
    for i in range(2):
        prefix = 'yaw' if i == 0 else 'pitch'
        poses = yaw_poses if i == 0 else pitch_poses
        for pose in poses:
            if i == 0:
                rotated_mesh, rotate_landmarks, original_angles, rotate_landmarks_106\
                    = render.rotate_render(data['param_path'], real_image, data['M'], yaw_pose=pose)
            else:
                rotated_mesh, rotate_landmarks, original_angles, rotate_landmarks_106\
                    = render.rotate_render(data['param_path'], real_image, data['M'], pitch_pose=pose)
            rotated_meshs.append(rotated_mesh)
            rotated_landmarks_list.append(rotate_landmarks)
            rotated_landmarks_list_106.append(rotate_landmarks_106)
            original_angles_list.append(original_angles)
            paths += data['path']
            pose_list += ['{}_{}'.format(prefix, pose) for i in range(len(data['path']))]
            real_images.append(real_image)
    rotated_meshs = torch.cat(rotated_meshs, 0)
    rotated_landmarks_list = torch.cat(rotated_landmarks_list, 0)
    rotated_landmarks_list_106 = torch.cat(rotated_landmarks_list_106, 0)
    original_angles_list = torch.cat(original_angles_list, 0)
    output = {}
    real_image = real_image * 2 - 1
    rotated_meshs = rotated_meshs * 2 - 1
    output['image'] = real_image.cpu()
    output['rotated_mesh'] = rotated_meshs.cpu()
    output['rotated_landmarks'] = rotated_landmarks_list.cpu()
    output['rotated_landmarks_106'] = rotated_landmarks_list_106.cpu()
    output['original_angles'] = original_angles_list.cpu()
    output['path'] = paths
    output['pose_list'] = pose_list
    return output