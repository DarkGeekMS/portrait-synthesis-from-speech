import os
import cv2
import json
import argparse
import numpy as np

def filter_matches(json_file):
    data = []
    for line in open(json_file, 'r'):
        data.append(json.loads(line))

    selection = dict()    
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
        cv2.imshow("Current Sample", complete_img)
        cv2.waitKey(0)
        select_idx = int(input("Enter index of best match [1|2|3|4|5]"))
        selection[sample['img_path'].split('/')[-1]] = sample['sim_imgs_paths'][select_idx-1].split('/')[-1]
        count += 1

    with open('final_match.json', 'w') as outfile:
        json.dump(selection, outfile)

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-jf', '--json_file', type=str, help='path to JSON containing faces matches')

    args = argparser.parse_args()

    filter_matches(args.json_file)
