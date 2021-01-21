from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os

class1_list = os.listdir('class-1')
class1_list = [os.path.join('class-1', npy_file) for npy_file in class1_list if npy_file[-4:]=='.npy']

class2_list = os.listdir('class-2')
class2_list = [os.path.join('class-2', npy_file) for npy_file in class2_list if npy_file[-4:]=='.npy']

xtrain = list()
ytrain = list()
for file in class1_list:
    xtrain.append(np.reshape(np.load(file), (18*512,)))
    ytrain.append(0)
for file in class2_list:
    xtrain.append(np.reshape(np.load(file), (18*512,)))
    ytrain.append(1)

clf = LogisticRegression(class_weight='balanced').fit(xtrain, ytrain)
direction = clf.coef_.reshape((18, 512))
np.save('skin_color.npy', direction)
