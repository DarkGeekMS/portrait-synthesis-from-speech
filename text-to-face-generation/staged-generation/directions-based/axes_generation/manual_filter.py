import os
import shutil
import cv2
import numpy as np

img_list = os.listdir('images')
img_list = [os.path.join('images', img_file) for img_file in img_list if img_file[-4:]=='.png']

count = 1
for img_file in img_list:
    orig_img = cv2.resize(cv2.imread(img_file), (256,256))
    while(1):
        cv2.imshow("Current Sample", orig_img)
        select_idx = cv2.waitKey(33)
        if select_idx == ord('a'):
            shutil.copy(img_file, f'class-1/{count}.png')
            shutil.copy(img_file[:-4]+'.npy', f'class-1/{count}.npy')
            break
        elif select_idx == ord('d'):
            shutil.copy(img_file, f'class-2/{count}.png')
            shutil.copy(img_file[:-4]+'.npy', f'class-2/{count}.npy')
            break
    count += 1
