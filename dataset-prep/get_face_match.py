from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
from shutil import copy2
import shutil
import numpy as np
import json
from scipy import spatial
import progressbar
import argparse
import sys

def get_annotated_samples(text_json, target_faces_dir):
    # extract the annotated faces from full dataset
    sub_dir = os.path.join(os.getcwd(), 'sub-faces')
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    else:
        shutil.rmtree(sub_dir)
        os.makedirs(sub_dir)
    with open(text_json) as json_file:
        data = json.load(json_file)
        for p in data:
            src_path = target_faces_dir + p['filename']
            dst_path = sub_dir + p['filename']
            copy2(src_path, dst_path)

def extract_database_embeds(faces_db_dir):
    # extract the embeddings from faces database to JSON file
    # if required, create a face detection pipeline using MTCNN
    mtcnn = MTCNN(image_size=256, margin=120)
    # create an inception resnet (in eval mode)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    paths = os.listdir(faces_db_dir)
    # create progress bar
    bar = progressbar.ProgressBar(maxval=len(paths), \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # extract embeddings
    embeds = list()
    count = 0
    for p in paths:
        path = os.path.join(faces_db_dir, p)
        resnet.classify = False
        # open the image
        img = Image.open(path)
        # get cropped and prewhitened image tensor
        try:
            img_cropped = mtcnn(img, save_path=None)
        except:
            print(f'{path} does not contain a clear face!')
            continue
        if img_cropped != None:
            # calculate embedding (unsqueeze to add batch dimension)
            img_embedding = resnet(img_cropped.unsqueeze(0))[0]
            bar.update(count + 1)
            embeds.append({
                'embedding':img_embedding.detach().numpy().tolist(),
                'path':path,
            })
        count += 1
    bar.finish()
    return embeds

def euclidean(a, b):
    # calculate euclidean distance
    a = np.array(a)
    b = np.array(b)
    return (sum((a-b)**2))** .5

def cosine_sim(a, b):
    # calculate cosine similarity
    return 1 - spatial.distance.cosine(a, b)

def get_sim_imgs(mtcnn, resnet, img_path, k, data):
    # get similar faces to a specific face
    img = Image.open(img_path)
    # get cropped and prewhitened image tensor
    try:
        img_cropped = mtcnn(img, save_path=None)
    except:
        print(f'{img_path} does not contain a clear face!')
        return
    # calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0)).detach().numpy().tolist()[0]
    sim_imgs = []
    # calculate all euclidean distance
    for d in data:
        d['euc'] = euclidean(img_embedding, d['embedding'])
    # get k minimum distances
    for i in range(k):
        idx = min(range(len(data)), key=lambda index: data[index]['euc'])
        sim_imgs.append(data[idx])
        del data[idx]
    # prepare outputs
    out = []
    for i in sim_imgs:
        out.append(i['path'])
    return out

def get_all_dataset_sim_imgs(k, target_faces_dir, embed_data):
    # get similar faces to all faces dataset
    # if required, create a face detection pipeline using MTCNN
    mtcnn = MTCNN(image_size=256, margin=120)
    # create an inception resnet (in eval mode)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    # prepare face images path
    included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
    img_paths = [target_faces_dir + '/' + img_name for img_name in os.listdir(target_faces_dir)
                                                 if any(img_name.endswith(ext) for ext in included_extensions)]
    # output json file path 
    out_json_path = './output_sim.json'
    # remove it if exists
    if(os.path.exists(out_json_path)):
        os.remove(out_json_path)
    # create progress bar
    bar = progressbar.ProgressBar(maxval=len(img_paths), \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    # get similarities
    count = 0
    for img_path in img_paths:
        sim_imgs_paths = get_sim_imgs(mtcnn, resnet, img_path, k, embed_data)
        if sim_imgs_paths is None:
            continue
        d = {
                'img_path':img_path,
                'sim_imgs_paths':sim_imgs_paths,
            }
        with open(out_json_path, 'a') as outfile:
            json.dump(d, outfile)
            outfile.write('\n')
        bar.update(count + 1)
        count += 1
    bar.finish()

def main(text_json, target_faces_dir, faces_db_dir, sim_num):
    # main driver
    get_annotated_samples(text_json, target_faces_dir)
    embed_data = extract_database_embeds(faces_db_dir)
    get_all_dataset_sim_imgs(sim_num, target_faces_dir, embed_data)

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-tj', '--text_json', type=str, help='path to JSON containing text descriptions')
    argparser.add_argument('-fdi', '--faces_dir', type=str, help='directory containing faces to be matched')
    argparser.add_argument('-fdb', '--faces_database', type=str, help='directory of faces database to get matches from')
    argparser.add_argument('-k', '--sim_num', type=int, help='number of similar images to the target', default=5)

    args = argparser.parse_args()

    main(args.text_json, args.faces_dir, args.faces_database, args.sim_num)
