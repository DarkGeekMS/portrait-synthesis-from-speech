"""Extract embeddings from face images using FaceNet"""

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
from shutil import copy2
import shutil
import numpy as np
import json
import progressbar
import sys
import argparse

def process_data(dataset_path):
    # process dataset of faces to extract embeddings
    dataset_folder_name = dataset_path.split('/')[-1]

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=256, margin=120)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    paths = os.listdir(dataset_path)

    dataset_folder_name = dataset_path.split('/')[-1]

    # remove contents of json file
    if os.path.exists(dataset_folder_name + '.json'):
        f = open(dataset_folder_name + '.json',"w")
        f.close()

    bar = progressbar.ProgressBar(maxval=len(paths), \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()

    count = 0

    npy_folder_path = './facenet-processed-npys'

    if os.path.exists(npy_folder_path):
        shutil.rmtree(npy_folder_path)

    os.makedirs(npy_folder_path)

    for p in paths:
        path = dataset_path + '/' + p

        try :
            resnet.classify = False
            # open the image
            img = Image.open(path)
            # Get cropped and prewhitened image tensor
            img_cropped = mtcnn(img, save_path=None)

            if img_cropped != None:
                # Calculate embedding (unsqueeze to add batch dimension)
                img_embedding = resnet(img_cropped.unsqueeze(0))[0]
                bar.update(count + 1)

                d = {
                    'embedding':img_embedding.detach().numpy().tolist(),
                    'path':path,
                }
                with open(dataset_folder_name + '.json', 'a') as outfile:
                    json.dump(d, outfile)
                    outfile.write('\n')

                # print((path[2:].split('.')[0]).split('/')[-1])
                npy_path = npy_folder_path + '/' + (path[2:].split('.')[0]).split('/')[-1] + '.npy'
                # print(npy_path)
                np.save(npy_path, np.array(d['embedding']))
        
            count += 1
        except Exception as e :
            print(e)
            continue
    
    bar.finish()

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-facedir', '--faces_dir', type=str, help='directory containing faces to be described using facenet')
    args = argparser.parse_args()
    process_data(dataset_path = args.faces_dir)
