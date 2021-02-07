from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import argparse

def fit_axis(class1_dir, class2_dir, direction_name):
    # fit facial attribute axis (direction)
    # list class 1 samples
    class1_list = os.listdir(class1_dir)
    class1_list = [os.path.join(class1_dir, npy_file) for npy_file in class1_list if npy_file[-4:]=='.npy']
    # list class 2 samples
    class2_list = os.listdir(class2_dir)
    class2_list = [os.path.join(class2_dir, npy_file) for npy_file in class2_list if npy_file[-4:]=='.npy']
    # read class 1 and 2 samples
    xtrain = list()
    ytrain = list()
    for file in class1_list:
        xtrain.append(np.reshape(np.load(file), (18*512,)))
        ytrain.append(0)
    for file in class2_list:
        xtrain.append(np.reshape(np.load(file), (18*512,)))
        ytrain.append(1)
    # fit logistic regression model
    clf = LogisticRegression(class_weight='balanced').fit(xtrain, ytrain)
    # extract direction
    direction = clf.coef_.reshape((18, 512))
    # convert into unit direction
    new_direction = np.zeros(direction.shape)
    for layer in range(direction.shape[0]):
        new_direction[layer] = np.divide(direction[layer], np.sqrt(np.dot(direction[layer], direction[layer])))
    # save direction
    np.save(f'{direction_name}.npy', new_direction)

if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-c1d', '--class1_dir', type=str, help='path to class 1 directory', default='class-1')
    argparser.add_argument('-c2d', '--class2_dir', type=str, help='path to class 2 directory', default='class-2')
    argparser.add_argument('-dn', '--direction_name', type=str, help='name of generated direction', default='new_direction')

    args = argparser.parse_args()

    # call axis fit
    fit_axis(args.class1_dir, args.class2_dir, args.direction_name)
