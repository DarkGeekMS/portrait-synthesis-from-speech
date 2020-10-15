import os
import cv2
import json
import argparse
import numpy as np

def visualize_matches(json_file):
    data = []
    for line in open(json_file, 'r'):
        data.append(json.loads(line))

    count = 0
    for sample in data:
        orig_img = cv2.resize(cv2.imread(sample['img_path']), (256,256))
        sim_img_1 = cv2.resize(cv2.imread(sample['sim_imgs_paths'][0]), (256,256))
        sim_img_2 = cv2.resize(cv2.imread(sample['sim_imgs_paths'][1]), (256,256))
        sim_img_3 = cv2.resize(cv2.imread(sample['sim_imgs_paths'][2]), (256,256))
        sim_img_4 = cv2.resize(cv2.imread(sample['sim_imgs_paths'][3]), (256,256))
        sim_img_5 = cv2.resize(cv2.imread(sample['sim_imgs_paths'][4]), (256,256))
        upper_row = np.concatenate([orig_img, sim_img_1, sim_img_2], axis=1)
        lower_row = np.concatenate([sim_img_3, sim_img_4, sim_img_5], axis=1)
        complete_img = np.concatenate([upper_row, lower_row], axis=0)
        cv2.imwrite(f"viz/{count}.png", complete_img)
        count += 1

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-jf', '--json_file', type=str, help='path to JSON containing faces matches')

    args = argparser.parse_args()

    visualize_matches(args.json_file)
