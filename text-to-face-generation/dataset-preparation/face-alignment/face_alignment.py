
##### imports
import cv2
import copy
import math
import numpy as np
import png
import os
from skimage import io
import timeit
import api
from PIL import Image
import shutil
import argparse


def main(root_path):

    # create a folder for aligned faces
    if not os.path.exists('./aligned'):
        os.makedirs('./aligned')

    # list all folders in root
    folders = os.listdir(root_path)
    folders_paths = [os.path.join(root_path, f) for f in folders]

    for i in range(len(folders_paths)):
        folder_path = folders_paths[i]
        # create an empty folder in aligned directory for each folder in the root
        if os.path.exists(os.path.join('./aligned', folders[i])):
            shutil.rmtree(os.path.join('./aligned', folders[i]))
        os.makedirs(os.path.join('./aligned', folders[i]))
        
        # list all images in the folder
        images_paths = os.listdir(folder_path)
        for p in images_paths:
            # get src and dst path for each image
            src_path = os.path.join(folder_path, p)
            dst_path = os.path.join('./aligned', folders[i], p)

            # load the image from src
            try:
                img2_r = io.imread(src_path)
            except:
                # not an image
                continue

            # align the image
            try:
                img2_a = api.face_alignment(img2_r, scale = 1.05)
                new_img = np.array(img2_a[0])
            except:
                # failure -> no faces found
                print('img failed:', src_path)
                continue
            # save the image in dst
            new_img = Image.fromarray(new_img)
            new_img.save(dst_path)

        print('done folder:', folders[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-root', '--root_path', type = str, help = 'path to root directory that contains folders each contains images to be aligned', default = '../simple_images')
    args = parser.parse_args()
    main(args.root_path)

