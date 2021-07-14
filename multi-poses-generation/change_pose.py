#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import os
import math
from tqdm import tqdm
import time
import face_alignment
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors, get_aligned_param, get_5lmk_from_68lmk
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.params import param_mean, param_std
from utils.render import get_depths_image, cget_depths_image, cpncc, crender_colors
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as multiprocessing
from models.networks.sync_batchnorm import DataParallelWithCallback
import sys
import data
from util.iter_counter import IterationCounter
from options.test_options import TestOptions
from models.test_model import TestModel
from util.visualizer import Visualizer
from util import html, util
from torch.multiprocessing import Process, Queue, Pool
from data.data_utils import init_parallel_jobs
from skimage import transform as trans
import time
from models.networks.rotate_render import TestRender
import math
import matplotlib.pyplot as plt

multiprocessing.set_start_method('spawn', force=True)


STD_SIZE = 120

def create_path(a_path, b_path):
    name_id_path = os.path.join(a_path, b_path)
    if not os.path.exists(name_id_path):
        os.makedirs(name_id_path)
    return name_id_path


def create_paths(save_path, img_path, foldername='orig', folderlevel=2, pose='0'):
    save_rotated_path_name = create_path(save_path, foldername)

    path_split = img_path.split('/')
    rotated_file_savepath = save_rotated_path_name
    for level in range(len(path_split) - folderlevel, len(path_split)):
        file_name = path_split[level]
        if level == len(path_split) - 1:
            file_name = str(pose) + '_' + file_name
        rotated_file_savepath = os.path.join(rotated_file_savepath, file_name)
    return rotated_file_savepath

def affine_align(img, landmark=None, **kwargs):
    M = None
    src = np.array([
     [38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041] ], dtype=np.float32 )
    src=src * 224 / 112

    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    warped = cv2.warpAffine(img, M, (224, 224), borderValue = 0.0)
    return warped

def landmark_68_to_5(t68):
    le = t68[36:42, :].mean(axis=0, keepdims=True)
    re = t68[42:48, :].mean(axis=0, keepdims=True)
    no = t68[31:32, :]
    lm = t68[48:49, :]
    rm = t68[54:55, :]
    t5 = np.concatenate([le, re, no, lm, rm], axis=0)
    t5 = t5.reshape(10)
    return t5


def save_img(img, save_path):
    image_numpy = util.tensor2im(img)
    util.save_image(image_numpy, save_path, create_dir=True)
    return image_numpy

def generate_3d_model(img):
    # 1. load pre-tained model
    checkpoint_fp = 'phase1_pdc.pth.tar'
    arch = 'mobilenet_1'

    landmark_list = []

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    cudnn.benchmark = True
    model = model.cuda()
    model.eval()

    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    # 2. parse images list 

    alignment_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    img_ori = img

    cv2.imwrite("face_data/Images/target.jpg", img)

    pts_res = []
    Ps = []  # Camera matrix collection
    poses = []  # pose collection, [todo: validate it]
    vertices_lst = []  # store multiple face vertices
    ind = 0

    # face alignment model use RGB as input, result is a tuple with landmarks and boxes
    preds = alignment_model.get_landmarks(img_ori[:, :, ::-1])
    pts_2d_68 = preds[0]
    pts_2d_5 = get_5lmk_from_68lmk(pts_2d_68)
    landmark_list.append(pts_2d_5)
    roi_box = parse_roi_box_from_landmark(pts_2d_68.T)

    img = crop_img(img_ori, roi_box)
    # import pdb; pdb.set_trace()


    # forward: one step
    img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
    input = transform(img).unsqueeze(0)
    with torch.no_grad():
        input = input.cuda()
        param = model(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

    # 68 pts
    pts68 = predict_68pts(param, roi_box)

    roi_box = parse_roi_box_from_landmark(pts68)
    img_step2 = crop_img(img_ori, roi_box)
    img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
    input = transform(img_step2).unsqueeze(0)
    with torch.no_grad():
        input = input.cuda()
        param = model(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

    pts68 = predict_68pts(param, roi_box)

    pts_res.append(pts68)
    P, pose = parse_pose(param)
    Ps.append(P)
    poses.append(pose)

    # dense face 3d vertices
    vertices = predict_dense(param, roi_box)

    wfp_2d_img = "target.png"
    colors = get_colors(img_ori, vertices)
    # aligned_param = get_aligned_param(param)
    # vertices_aligned = predict_dense(aligned_param, roi_box)
    # h, w, c = 120, 120, 3
    h, w, c = img_ori.shape
    img_2d = crender_colors(vertices.T, (tri - 1).T, colors[:, ::-1], h, w)
    cv2.imwrite(wfp_2d_img, img_2d[:, :, ::-1])

    #Export parameters

    save_name = "face_data/params/target.txt"
    this_param = param * param_std + param_mean
    this_param = np.concatenate((this_param, roi_box))
    this_param.tofile(save_name, sep=' ')

    # #Export landmarks

    save_path = "face_data/realign_lmk"

    with open(save_path, 'w') as f:
        # f.write('{} {} {} {}')
        land = np.array(landmark_list[0])
        land = land.astype(np.int)
        land_str = ' '.join([str(x) for x in land])
        msg = f'target.jpg 0 {land_str}\n'
        f.write(msg)


def rotate_face(img, angle):

    generate_3d_model(img)

    opt = TestOptions()
    opt = opt.parse()

    opt.names = "rs_model"
    opt.dataset = "example" 
    opt.list_start = 0
    opt.list_end = 0
    opt.dataset_mode = "allface" 
    opt.gpu_ids = 0
    opt.netG = "rotatespade"
    opt.norm_G = "spectralsyncbatch"
    opt.model = "rotatespade" 
    opt.label_nc = 5
    opt.nThreads = 3
    opt.heatmap_size = 2.5
    opt.chunk_size = 1
    opt.no_gaussian_landmark = True
    opt.device_count = 1
    opt.render_thread = 1
    opt.label_mask = True
    opt.align = True
    opt.erode_kernel = 21 
    opt.yaw_poses = [math.radians(angle)]
    opt.batchSize=1
    opt.isTrain = False

    import data

    data_info = data.dataset_info()
    datanum = data_info.get_dataset(opt)[0]
    folderlevel = data_info.folder_level[datanum]

    dataloaders = data.create_dataloader_test(opt)

    visualizer = Visualizer(opt)
    iter_counter = IterationCounter(opt, len(dataloaders[0]) * opt.render_thread)
    # create a webpage that summarizes the all results

    testing_queue = Queue(10)

    ngpus = opt.device_count

    render_gpu_ids = list(range(ngpus - opt.render_thread, ngpus))
    render_layer_list = []
    for gpu in render_gpu_ids:
        opt.gpu_ids = gpu
        render_layer = TestRender(opt)
        render_layer_list.append(render_layer)

    opt.gpu_ids = [0]
    print('Testing gpu ', opt.gpu_ids)
    if opt.names is None:
        model = TestModel(opt)
        model.eval()
        model = torch.nn.DataParallel(model.cuda(),
                                      device_ids=opt.gpu_ids,
                                      output_device=opt.gpu_ids[-1],
                                      )
        models = [model]
        names = [opt.name]
        save_path = create_path(create_path(opt.save_path, opt.name), opt.dataset)
        save_paths = [save_path]
        f = [open(
                os.path.join(save_path, opt.dataset + str(opt.list_start) + str(opt.list_end) + '_rotate_lmk.txt'), 'w')]
    else:
        models = []
        names = []
        save_paths = []
        f = []
        for name in opt.names.split(','):
            opt.name = name
            model = TestModel(opt)
            model.eval()
            model = torch.nn.DataParallel(model.cuda(),
                                          device_ids=opt.gpu_ids,
                                          output_device=opt.gpu_ids[-1],
                                          )
            models.append(model)
            names.append(name)
            save_path = create_path(create_path(opt.save_path, opt.name), opt.dataset)
            save_paths.append(save_path)
            f_rotated = open(
                os.path.join(save_path, opt.dataset + str(opt.list_start) + str(opt.list_end) + '_rotate_lmk.txt'), 'w')
            f.append(f_rotated)

    test_tasks = init_parallel_jobs(testing_queue, dataloaders, iter_counter, opt, render_layer_list)
    # test
    landmarks = []

    process_num = opt.list_start
    first_time = time.time()
    try:
        for i, data_i in enumerate(range(len(dataloaders[0]) * opt.render_thread)):
            # if i * opt.batchSize >= opt.how_many:
            #     break
            # data = trainer.get_input(data_i)
            start_time = time.time()
            data = testing_queue.get(block=True)

            current_time = time.time()
            time_per_iter = (current_time - start_time) / opt.batchSize
            message = '(************* each image render time: %.3f *****************) ' % (time_per_iter)
            print(message)

            img_path = data['path']
            poses = data['pose_list']
            rotated_landmarks = data['rotated_landmarks'][:, :, :2].cpu().numpy().astype(np.float)
            rotated_landmarks_106 = data['rotated_landmarks_106'][:, :, :2].cpu().numpy().astype(np.float)


            generate_rotateds = []
            for model in models:
                generate_rotated = model.forward(data, mode='single')
                generate_rotateds.append(generate_rotated)

            for n, name in enumerate(names):
                opt.name = name
                for b in range(generate_rotateds[n].shape[0]):
                    # get 5 key points
                    rotated_keypoints = landmark_68_to_5(rotated_landmarks[b])
                    # get savepaths
                    rotated_file_savepath = create_paths(save_paths[n], img_path[b], folderlevel=folderlevel, pose=poses[b])

                    image_numpy = save_img(generate_rotateds[n][b], rotated_file_savepath)
                    rotated_keypoints_str = rotated_file_savepath + ' 1 ' + ' '.join([str(int(n)) for n in rotated_keypoints]) + '\n'
                    print('process image...' + rotated_file_savepath)
                    f[n].write(rotated_keypoints_str)

                    current_time = time.time()
                    if n == 0:
                        if b <= opt.batchSize:
                            process_num += 1
                        print('processed num ' + str(process_num))
                    if opt.align:
                        aligned_file_savepath = create_paths(save_paths[n], img_path[b], 'aligned', folderlevel=folderlevel, pose=poses[b])
                        warped = affine_align(image_numpy, rotated_keypoints.reshape(5, 2))
                        util.save_image(warped, aligned_file_savepath, create_dir=True)

                    # save 106 landmarks
                    rotated_keypoints_106 = rotated_landmarks_106[b] # shape: 106 * 2


            current_time = time.time()
            time_per_iter = (current_time - start_time) / opt.batchSize
            message = '(************* each image time total: %.3f *****************) ' % (time_per_iter)
            print(message)

            return warped

    except KeyboardInterrupt:
        print("Interrupted!")
        for fs in f:
            fs.close()
        pass

    except Exception as e:
        print(e)
        for fs in f:
            fs.close()

    else:
        print('finished')
        for fs in f:
            fs.close()

if __name__ == '__main__':
    img = cv2.imread("sebak.jpg")
    rotated = rotate_face(img,0)
    plt.imshow(rotated)
    plt.show()


