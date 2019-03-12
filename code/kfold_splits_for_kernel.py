from pandas import read_csv
from os.path import isfile
#from PIL import Image as pil_image
import pickle
import numpy as np
#from imagehash import phash
from math import sqrt
import gzip
import os 
from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras import backend as K 
from keras.utils import Sequence
import random 
from lap import lapjv
import time
import math  
import pandas as pd 
from sklearn.model_selection import KFold
from a02_model_modifications import * 

# Suppress annoying stderr output when importing keras.
import sys
import platform
old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')
import keras
sys.stderr = old_stderr

import random
from keras import backend as K
from keras.preprocessing.image import img_to_array,array_to_img
from scipy.ndimage import affine_transform
from multiprocessing.pool import ThreadPool
from functools import partial
import operator

from contextlib import contextmanager

########################################################
################# data preparation #####################
########################################################


### using bbox 
temp_p2bb = pickle.load(open('../modified_data/p2bb_averaged_v1.pkl', 'rb'))
p2bb1 = {}
for k in temp_p2bb:
    p2bb1[k+'.jpg'] = temp_p2bb[k]

temp_p2bb = pickle.load(open('../modified_data/p2bb_averaged_playground_v1.pkl', 'rb'))
p2bb2 = {}
for k in temp_p2bb:
    p2bb2[k+'.jpg'] = temp_p2bb[k]
p2bb = {**p2bb1, **p2bb2}

# load tagged data
tagged = dict([(p[:-4]+'.jpg',w) for _,p,w in read_csv('../data/train.csv').to_records()])
#newdata = dict([(k[:-4]+'.jpg', v) for k,v in pickle.load(open('../modified_data/new_train_data.pkl', 'rb')).items()])
#newdata_bootstrap_Jan30 = dict([(k[:-4]+'.jpg', v) for k,v in pickle.load(open('../modified_data/bootstrap_data/pseudo_label_test_data_LB965.pkl', 'rb')).items()])
#tagged = dict(**tagged, **newdata, **newdata_bootstrap_Jan30)

# whale ID -> photo
w2p = {} 
for img, lb in tagged.items():
    if img in p2bb:      # remove bad images!
        if lb not in w2p:
            w2p[lb] = [img]
        else:
            w2p[lb].append(img)

########################################################
################### CV preparation #####################
########################################################

new_kfolds = {}
new_whale_perc = 0.276
TOTAL_FOLD = 4 

global_seed = 0 

# shuffle all data 
np.random.seed(100)
for k in w2p:
    np.random.shuffle(w2p[k])

# prepare new_whale data 
valid_neg = []
for k in w2p:
    if k == 'new_whale':
        valid_neg += w2p[k]
valid_neg_batch = len(valid_neg) // TOTAL_FOLD
np.random.seed(1000)
np.random.shuffle(valid_neg)

# shuffle validation fold indices for classes >= 4 images 
val_fold_index = {}
np.random.seed(10000)
for k in w2p:
    if len(w2p[k]) > 3:
        val_fold_index[k] = np.random.choice([0,1,2,3], 4, replace=False)


def find_train_val_split_4_fold(k, l, run_fold):
    assert len(l) > 1
    kf = KFold(n_splits=4)

    if len(l) == 2:
        return l, []
    elif len(l) == 3:
        if run_fold == 3:
            global global_seed
            global_seed += 1 
            #print("Global seed: ", global_seed)
            np.random.seed(global_seed)
            #val_img = np.random.choice(l, 1)
            np.random.shuffle(l)
            #print(l[1:], l[0])
            return l[1:], [l[0]]
        else:
            return l[:run_fold]+l[run_fold+1:], [l[run_fold]]
    else:
        four_fold_indices = val_fold_index[k] 
        shuffled_fold_index = four_fold_indices[run_fold]
        tr, val = list(kf.split(l))[shuffled_fold_index]
        return [l[i] for i in tr], [l[i] for i in val]


for RUN_FOLD in range(4):

    train, val = [], []
    w2ts = {} # valid whale label to photos

    for k in w2p:
        if len(w2p[k]) > 1 and k != 'new_whale':
            train_split, val_split = find_train_val_split_4_fold(k, w2p[k], RUN_FOLD)
            train += train_split
            val += val_split
            w2ts[k] = np.array(train_split).astype(object)

    train = sorted(train)

    new_whale_for_val = int(len(val) / (1-new_whale_perc) * new_whale_perc)
    #print("new_whale needed for this fold: ", new_whale_for_val)

    #print("Total new whale images we have: ", len(valid_neg))
    valid_neg_fold = valid_neg[RUN_FOLD*valid_neg_batch:(RUN_FOLD+1)*valid_neg_batch][:new_whale_for_val]   
    #print("Total new_whale in val for this fold: ", len(valid_neg_fold))

    val += valid_neg_fold
    print("\nTotal train images: ", len(train))
    print("\nTotal val images: {}\n".format(len(val)))

    new_kfolds['fold_{}'.format(RUN_FOLD)] = {}
    new_kfolds['fold_{}'.format(RUN_FOLD)]['train'] = train
    new_kfolds['fold_{}'.format(RUN_FOLD)]['val'] = val

# double check if kfold split is correct - no overlap between train and val 
for fd in new_kfolds.keys():
    f_train, f_val = new_kfolds[fd]['train'], new_kfolds[fd]['val']
    total_n = len(f_train) + len(f_val)
    total_set = len(set(f_train + f_val))
    assert total_n == total_set

pickle.dump(new_kfolds, open('../modified_data/new_4_folds_split_train_val.pkl', 'wb'))


    