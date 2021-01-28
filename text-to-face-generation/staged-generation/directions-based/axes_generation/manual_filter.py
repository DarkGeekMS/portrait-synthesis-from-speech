import os
import shutil
import cv2
import argparse

def manual_filter(img_dir, class1_dir, class2_dir):
    # filter generated images manually
    # read images from given directory
    img_list = os.listdir(img_dir)
    img_list = [os.path.join(img_dir, img_file) for img_file in img_list if img_file[-4:]=='.png']
    # loop over each image
    count = 1
    for img_file in img_list:
        # read image
        orig_img = cv2.resize(cv2.imread(img_file), (512,512))
        while(1):
            # display image
            cv2.imshow(f"Current Sample", orig_img)
            # get selected class
            select_idx = cv2.waitKey(33)
            if select_idx == ord('a'):
                shutil.copy(img_file, f'{class1_dir}/{count}.png')
                shutil.copy(img_file[:-4]+'.npy', f'{class1_dir}/{count}.npy')
                break
            elif select_idx == ord('d'):
                shutil.copy(img_file, f'{class2_dir}/{count}.png')
                shutil.copy(img_file[:-4]+'.npy', f'{class2_dir}/{count}.npy')
                break
        count += 1

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-id', '--img_dir', type=str, help='path to geenrated images directory', default='images')
    argparser.add_argument('-c1d', '--class1_dir', type=str, help='path to class 1 directory', default='class-1')
    argparser.add_argument('-c2d', '--class2_dir', type=str, help='path to class 2 directory', default='class-2')

    args = argparser.parse_args()

    # create output directories
    if not os.path.isdir(args.class1_dir):
        os.mkdir(args.class1_dir)

    if not os.path.isdir(args.class2_dir):
        os.mkdir(args.class2_dir)

    # call manual filter
    manual_filter(args.img_dir, args.class1_dir, args.class2_dir)
