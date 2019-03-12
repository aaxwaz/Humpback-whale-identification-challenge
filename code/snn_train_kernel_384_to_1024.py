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
from keras.layers import Activation, Add, BatchNormalization, Dropout,Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras import backend as K 
from keras.utils import Sequence
import random 
from lap import lapjv
import time
import math  
import pandas as pd 
from sklearn.model_selection import KFold
from a01_adam_accumulate import AdamAccumulate
from keras.applications.inception_v3 import InceptionV3
from albumentations import * 
import cv2 
from a02_model_modifications import * 
from a01_adam_accumulate import * 
from a00_common_functions import * 
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
from keras.utils import multi_gpu_model
from PIL import Image as pil_image

from contextlib import contextmanager
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.1f} s')

import argparse

###########################################################
################# Set up basic config #####################
###########################################################

parser = argparse.ArgumentParser()
parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0,1", help="GPU index to use. Default 0,1. Note that it is required to use two GPUs for training with image size 1,024. ", type=str)
parser.add_argument("--RUN_FOLD", default=0, help="Fold to train in k-fold split training approach. Default 0. ", type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES   
print("\nUsing GPU: {}\n".format(os.environ["CUDA_VISIBLE_DEVICES"])) 

TOTAL_FOLD = 4
RUN_FOLD = args.RUN_FOLD

print("Using {} fold CV. Current fold: {}\n".format(TOTAL_FOLD, RUN_FOLD))

###########################################################
################# data preparation ########################
###########################################################

LINEAR_ASSIGNMENT_SEGMENT_SIZE = 2
segment = True
if segment:
    print("\nUsing LINEAR_ASSIGNMENT_SEGMENT_SIZE: {} \n\n".format(LINEAR_ASSIGNMENT_SEGMENT_SIZE))

#img_shape    = (262,int(262*2.15),3) # The image shape used by the model
anisotropy   = 2.15 # The horizontal compression ratio
crop_margin  = 0.05 # The margin added around the bounding box to compensate for bounding box inaccuracy
rotate = {}
exclude = []

def expand_path_RGB(p):
    if isfile('../data/train/' + p): return '../data/train/' + p
    if isfile('../data/test/' + p): return '../data/test/' + p
    return None

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

# whale ID -> photo
w2p = {} 
for img, lb in tagged.items():
    if img in p2bb:      # remove bad images!
        if lb not in w2p:
            w2p[lb] = [img]
        else:
            w2p[lb].append(img)

# test
submit = [p for _,p,_ in read_csv('../data/sample_submission.csv').to_records()]
submit = [f[:-4]+'.jpg' for f in submit]
join   = set(list(tagged.keys()) + submit)
print(len(tagged),len(submit),len(join),list(tagged.items())[:5],submit[:5])


########################################################
################### CV preparation #####################
########################################################

splits = pickle.load(open('../modified_data/new_4_folds_split_train_val.pkl', 'rb'))

this_fold = splits['fold_{}'.format(RUN_FOLD)]
this_fold_train = np.array(this_fold['train'])
this_fold_val = np.array(this_fold['val'])

train, val = this_fold_train, this_fold_val

w2ts = {} # valid whale label to photos

for k in w2p:
    if len(w2p[k]) > 1 and k != 'new_whale':
        temp = []
        for p in w2p[k]:
            if p in train:
                temp.append(p)
        w2ts[k] = temp 

print("\nTotal train and val for this fold:", len(train), len(val))

########################################################
################# training preparation #################
########################################################

"""def read_raw_image_npy(p):
    img = np.load(expand_path_RGB(p))
    if p in rotate: img = img.rotate(180)
    return img"""

def read_raw_image_npy(p):
    img = img_to_array(pil_image.open(expand_path_RGB(p))).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[-1] == 3:
        return img 
    elif len(img.shape) == 2:
        return np.stack((img,)*3, axis=-1)
    else:
        return np.stack((img[:,:,0],)*3, axis=-1)


def stronger_aug(p=.5):
    return Compose([

        #ElasticTransform(alpha=203, sigma=166, alpha_affine=106, p=0.9),

        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=5, p=0.2),

        HorizontalFlip(p=0.005),

        VerticalFlip(p=0.005),

        Rotate(limit=30, p = 0.7),
        
        RandomRotate90(p=0.02),

        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
            #ElasticTransform(alpha=203, sigma=166, alpha_affine=106, p=0.2), 
            RandomContrast(limit=0.9, p=0.9)
        ], p=0.5),

        OneOf([
            MotionBlur(p=0.5),
            MedianBlur(blur_limit=3, p=0.5),
            Blur(blur_limit=3, p=0.5),
        ], p=0.1),

        #CenterCrop(height=int(ori_image_size[0]/2*1.5), width=int(ori_image_size[1]/2*1.5), p=0.1), 
        #CenterCrop(height=512, width=512, p=0.1), 
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
        ], p=0.1),
        #ElasticTransform(alpha=30, sigma=30, alpha_affine=30, p=0.2), 
        OneOf([
            RGBShift(p=1.0, r_shift_limit=(-30, 30), g_shift_limit=(-30, 30), b_shift_limit=(-30, 30)),
            HueSaturationValue(p=1.0),
            RandomBrightness(limit=0.2, p =1.0)
        ], p=0.95),
        JpegCompression(p=0.4, quality_lower=20, quality_upper=99),
        # ElasticTransform(p=0.1),
        ToGray(p=0.3),
    ],
        bbox_params={'format': 'pascal_voc',
                        'min_area': 1,
                        'min_visibility': 0.1,
                        'label_fields': ['labels']},
    p=p)

def random_crop_no_resize(p, ori_image_size):
    crop_h = np.random.uniform(0.7, 0.9)
    crop_w = np.random.uniform(0.7, 0.9)
    return Compose([
        RandomCrop(height=int(ori_image_size[0]*crop_h), width=int(ori_image_size[1]*crop_w), p=p)], 
        #RandomSizedCrop(min_max_height=(int(ori_image_size[1]*0.2), int(ori_image_size[1]*0.2)), height=img_shape[1], width=img_shape[0], w2h_ratio = float(ori_image_size[0]/ori_image_size[1]*5), p=p)], 
    p=1.0)

def preprocess_image(img):
    img /= 127.5
    img -= 1.
    return img

def read_cropped_image_npy(p, augment):
    anisotropy = 2.15  # The horizontal compression ratio
    if augment:
        crop_margin = random.uniform(0.01, 0.09)
    else:
        crop_margin = 0.05

    x0, y0, x1, y1 = p2bb[p]
    # Read the image
    img = read_raw_image_npy(p)
    # print(img.shape, img.min(), img.max())
    # show_image(img[:, :, 0])
    size_x, size_y = img.shape[1], img.shape[0]

    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if x0 < 0: x0 = 0
    if x1 > size_x: x1 = size_x
    if y0 < 0: y0 = 0
    if y1 > size_y: y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * anisotropy:
        dy = 0.5 * (dx / anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * anisotropy - dx)
        x0 -= dx
        x1 += dx

    # img1 = cv2.rectangle(img.copy(), (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2, cv2.LINE_AA)
    # show_image(img1)

    if x0 < 0: x0 = 0
    if x1 > size_x: x1 = size_x
    if y0 < 0: y0 = 0
    if y1 > size_y: y1 = size_y
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    if augment:
        # print(x0, y0, x1, y1)
        bb = np.array([[x0, y0, x1, y1]])
        #try:
        #print("w2h_ratio: ", size_x/size_y)
        #print(img.shape)
        augm = stronger_aug(p=1.0)(image=img, bboxes=bb, labels=['labels'])
        #except:
        #    print('Error albumentations: {}'.format(os.path.basename(p)))
        #sys.exit()
        img = augm['image']
        bboxes = np.array(augm['bboxes'])[0]
        x0, y0, x1, y1 = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])
        # img2 = cv2.rectangle(img.copy(), (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2, cv2.LINE_AA)
        # show_image(img2)
        if x0 < 0: x0 = 0
        if x1 > size_x: x1 = size_x
        if y0 < 0: y0 = 0
        if y1 > size_y: y1 = size_y

    if y0 != y1 and x0 != x1:
        img = img[y0:y1, x0:x1, :]

    if augment:
        img = random_crop_no_resize(p = 0.5, ori_image_size=(img.shape[0], img.shape[1]))(image=img)['image']
    try:
        img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    except:
        print('Resize error! Image shape: {}'.format(img.shape), p, x0, y0, x1, y1)
        img = read_raw_image_npy(p)
        img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)

    if len(img.shape) == 2:
        img = np.concatenate((img, img, img), axis=2)

    #img = array_to_img(img)
    img = img.astype(np.float32)
    #img  -= np.mean(img, keepdims=True)
    #img  /= np.std(img, keepdims=True) + K.epsilon()
    img = preprocess_image(img)

    return img 



def read_for_training(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image_npy(p, True)

def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image_npy(p, False)

def prepare_val_res(score, threshold):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    new_whale = 'new_whale'

    res  ={}

    train_arr = np.array(train)

    for i,p in enumerate(val):
        t = []
        s = set()
        a = score[i,:]

        top_label_probs = {}
        cond = a > threshold
        cond_index = np.where(cond)[0]
        cond_images = train_arr[cond_index]
        for j, img in enumerate(cond_images):
            if tagged[img] in top_label_probs:
                top_label_probs[tagged[img]] += a[cond_index[j]]
            else:
                top_label_probs[tagged[img]] = a[cond_index[j]]

        sorted_top_label_probs = sort_dict_by_values(top_label_probs)

        t = []
        for lb, _ in sorted_top_label_probs:
            t.append(lb)

        if len(t) < 5:
            t.append(new_whale)

        for index in np.argsort(a)[::-1]:
            if tagged[train_arr[index]] not in t:
                t.append(tagged[train_arr[index]])
            if len(t) >= 5:
                break

        assert len(t) >= 5

        res[p[:-4]+'.jpg'] =  t[:5]

    return res 


class TrainingDataFast(Sequence):
    def __init__(self, score_list, steps=1000, batch_size=32):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingDataFast, self).__init__()
        #self.score      = -score # Maximizing the self.score is the same as minimuzing -score.
        self.score = np.zeros((len(train), len(train)))
        start = 0 
        for s in score_list:
            the_size = s.shape[0]
            self.score[start:start+the_size, start:start+the_size] = -s
            start += the_size

        self.steps      = steps
        self.batch_size = batch_size
        for ts in w2ts.values():
            idxs = [t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    self.score[i,j] = 10000.0 # Set a large value for matching whales -- eliminates this potential pairing

        self.on_epoch_end()
        self.p = ThreadPool(8)

    def __getitem__(self, index):
        start = self.batch_size*index
        end   = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size  = end - start
        assert size > 0

        def _read_for_training_expand_dim(p):
            return np.expand_dims(read_for_training(p), 0).astype(K.floatx())

        i_start, i_end = start//2, int(start//2+np.ceil(size/2))
        unroll_match = [[m[0], m[1], u[0], u[1]] for m, u in zip(self.match[i_start:i_end], self.unmatch[i_start:i_end])]
        unroll_match = [i for item in unroll_match for i in item]
        unroll_train_images = np.concatenate(self.p.map(_read_for_training_expand_dim, unroll_match), 0)
        #print("\nunroll_train_images shape: {}\n".format(unroll_train_images.shape))

        ## a[0], b[0], a[1], b[1], a[2], b[2], a[3], a[4] ...
        a_indices = list(range(0, len(unroll_train_images), 2))
        a = unroll_train_images[a_indices]
        b_indices = list(range(1, len(unroll_train_images), 2))
        b = unroll_train_images[b_indices]
        c = np.zeros((size,1), dtype=K.floatx())
        for i in range(0, size, 2):
            c[i,  0    ] = 1 # This is a match
            c[i+1,0    ] = 0 # Different whales

        return [a,b],c
    def on_epoch_end(self):
        if self.steps <= 0: return # Skip this on the last epoch.
        self.steps     -= 1
        self.match      = [] # len(self.match) == len(train), each element is a pair of images that match the same label 
        self.unmatch    = []
        if segment:
            # Using slow scipy. Make small batches.
            # Because algorithm is O(n^3), small batches are much faster.
            # However, this does not find the real optimum, just an approximation.
            tmp   = []
            batch = math.ceil(self.score.shape[0] / LINEAR_ASSIGNMENT_SEGMENT_SIZE)
            print("\nScore matrix size: {} and lapjv batch size: {}".format(self.score.shape[0], batch))
            for start in range(0, self.score.shape[0], batch):
                end = min(self.score.shape[0], start + batch)
                #_, x = linear_sum_assignment(self.score[start:end, start:end])
                #_, x = hungarian.lap(self.score[start:end, start:end])
                with timer("lapjv ops 1 round"):
                    _,_,x = lapjv(self.score[start:end, start:end])
                tmp.append(x + start)
            x = np.concatenate(tmp)
        else:
            print("Running lapjv on entire score matrix! size: ".format)
            with timer("lapjv ops (entire data) "):
                _,_,x = lapjv(self.score) # Solve the linear assignment problem

        y = np.arange(len(x),dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): break
            for ab in zip(ts,d): self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i,j in zip(x,y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i,j)
            assert i != j
            self.unmatch.append((train[i],train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x,y] = 10000.0
        self.score[y,x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(train) and len(self.unmatch) == len(train)
    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1)//self.batch_size

# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, batch_size=32, verbose=1):
        super(FeatureGen, self).__init__()
        self.data       = data
        self.batch_size = batch_size
        self.verbose    = verbose
        self.p = ThreadPool(8)
        #if self.verbose > 0: self.progress = tqdm_notebook(total=len(self), desc='Features')
        #if self.verbose > 0: self.progress = range(total=len(self))

    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.data) - start, self.batch_size)
        def _read_for_validation_expand_dim(p):
            return np.expand_dims(read_for_validation(p), 0).astype(K.floatx())
        res = self.p.map(_read_for_validation_expand_dim, self.data[start:start+size])
        res = np.concatenate(res, 0)

        return res

    def __len__(self):
        return (len(self.data) + self.batch_size - 1)//self.batch_size

# A Keras generator to evaluate on the HEAD MODEL on features already pre-computed.
# It computes only the upper triangular matrix of the cost matrix if y is None.
class ScoreGen(Sequence):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x          = x
        self.y          = y
        self.batch_size = batch_size
        self.verbose    = verbose
        if y is None:
            self.y           = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0],1)
        else:
            self.iy, self.ix = np.indices((y.shape[0],x.shape[0]))
            self.ix          = self.ix.reshape((self.ix.size,))
            self.iy          = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1)//self.batch_size
        #if self.verbose > 0: self.progress = tqdm_notebook(total=len(self), desc='Scores')
        #if self.verbose > 0: self.progress = range(total=len(self))
    def __getitem__(self, index):
        start = index*self.batch_size
        end   = min(start + self.batch_size, len(self.ix))
        a     = self.y[self.iy[start:end],:]
        b     = self.x[self.ix[start:end],:]
        #if self.verbose > 0: 
        #    self.progress.update()
        #    if self.progress.n >= len(self): self.progress.close()
        return [a,b]
    def __len__(self):
        return (len(self.ix) + self.batch_size - 1)//self.batch_size

def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))

def get_lr(model):
    return K.get_value(model.optimizer.lr)

def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0],x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0],1)] = score.squeeze()
        m += m.transpose()
    else:
        m        = np.zeros((y.shape[0],x.shape[0]), dtype=K.floatx())
        iy,ix    = np.indices((y.shape[0],x.shape[0]))
        ix       = ix.reshape((ix.size,))
        iy       = iy.reshape((iy.size,))
        m[iy,ix] = score.squeeze()
    return m

def compute_score_fast(verbose=1):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    res = []

    batch = math.ceil(len(train) / LINEAR_ASSIGNMENT_SEGMENT_SIZE)
    for start in range(0, len(train), batch):
        end = min(len(train), start + batch)
        train_batch = train[start:end]

        features = branch_model.predict_generator(FeatureGen(train_batch, verbose=verbose), max_queue_size=12, workers=6, verbose=0)
        score    = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6, verbose=0)
        score    = score_reshape(score, features)

        res.append(score)

    return res

def compute_fake_score():

    res = []

    batch = math.ceil(len(train) / LINEAR_ASSIGNMENT_SEGMENT_SIZE)
    for start in range(0, len(train), batch):
        end = min(len(train), start + batch)
        train_batch = train[start:end]

        res.append(np.zeros((end-start, end-start)))

    return res

def make_steps_fast(step, ampl, verbose=2, batch_size = 20, no_compute_score = False):
    """
    Perform training epochs
    @param step Number of epochs to perform`
    @param ampl the K, the randomized component of the score matrix.
    """
    global w2ts, t2i, steps, features, score, histories

    print("Using batch size of {} for training. ".format(batch_size))
    
    # shuffle the training pictures
    random.shuffle(train)
    
    # Map training picture hash value to index in 'train' array    
    t2i  = {}
    for i,t in enumerate(train): t2i[t] = i    

    # Compute the match score for each picture pair of train data - square matrix score[i,j] is the score between image i and j, 0~1
    if no_compute_score:
        print("Skipping computing score matrix. ")
        score_list = compute_fake_score()
    else:
        with timer("Compute score fast. "):
            score_list = compute_score_fast()    

    for i in range(len(score_list)):
        score_list[i] += ampl*np.random.random_sample(size=score_list[i].shape)
    
    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingDataFast(score_list, steps=step, batch_size=batch_size),
        initial_epoch=steps, epochs=steps + step, max_queue_size=12, workers=6, verbose=verbose,
        #callbacks=[
        #    TQDMNotebookCallback(leave_inner=True, metric_format='{value:0.3f}')
        #]
        ).history
    steps += step
    
    # Collect history data
    history['epochs'] = steps
    history['ms'    ] = np.mean(score)
    history['lr'    ] = get_lr(model)
    print(history['epochs'],history['lr'],history['ms'])
    histories.append(history)


def run_validation(threshold):
    print("\nRunning validation ...")
    ftrain  = branch_model.predict_generator(FeatureGen(train), max_queue_size=20, workers=10, verbose=0)
    fval = branch_model.predict_generator(FeatureGen(val), max_queue_size=20, workers=10, verbose=0)
    score   = head_model.predict_generator(ScoreGen(ftrain, fval), max_queue_size=20, workers=10, verbose=0)
    score   = score_reshape(score, ftrain, fval)

    res = prepare_val_res(score, threshold)

    ## calculate score
    scores = []
    missing = []
    for img in res.keys():
        if img in tagged:
            score = apk([tagged[img]], res[img], k=5)
            scores.append(score)
        else:
            print("Warning! val image: {} not in tagged. ".format(img))
            missing.append(img)

    print("\nValid score: {}".format(np.mean(np.array(scores))))

    return np.mean(np.array(scores))


########################################################
################# training starts ######################
########################################################

histories  = []
steps      = 0

saved_folder = 'PLAN21'
version = 'PLAN21_martinAsWarmStart_384to1024_fold_' + str(RUN_FOLD) 
pretrained_weights = '../modified_data/mpiotte-standard.model'
l2_rate = 0.0002

if not os.path.isdir('../saved_models'):
    os.makedirs('../saved_models')

if not os.path.isdir('../saved_models/{}'.format(saved_folder)):
    os.makedirs('../saved_models/{}'.format(saved_folder))
    
model_str = pretrained_weights.split('/')[-1].split('.')[0]

########### PLAN 12 procedures ###########
#run_validation(0.99)

score = np.random.random_sample(size=(len(train),len(train)))

if 1:
    img_shape = (384, 384, 3) 
    batch_size = 64
    
    model, branch_model, head_model = build_model_multich(3, pretrained_weights, img_shape)

    # 384 
    print("Weights loaded from: {}. Using img_shape: {}. Using batch_size: {}".format(pretrained_weights, img_shape, batch_size))
    best_score = run_validation(0.99)

    # epoch -> 30
    set_lr(model, 8e-5)
    print("\n{} training block 0 {}".format('*'*17, '*'*17))
    make_steps_fast(10, 1, batch_size=batch_size)
    model.save_weights('../saved_models/{}/{}_{}_temp0.model_weights'.format(saved_folder, model_str, version))
    print("model saved: ", '../saved_models/{}/{}_{}_temp0.model_weights'.format(saved_folder, model_str, version))


    set_lr(model, 8e-5)
    print("\n{} training block 1 {}".format('*'*17, '*'*17))
    make_steps_fast(20, 1, batch_size=batch_size)
    model.save_weights('../saved_models/{}/{}_{}_temp1.model_weights'.format(saved_folder, model_str, version))
    print("model saved: ", '../saved_models/{}/{}_{}_temp1.model_weights'.format(saved_folder, model_str, version))

    best_score = run_validation(0.99)


    # epoch -> 25
    this_weights = '../saved_models/{}/{}_{}_temp2.model_weights'.format(saved_folder, model_str, version)
    set_lr(model, 4e-5)
    print("\n\n{} training block 2 {}".format('*'*17, '*'*17))
    for _ in range(5): 
        make_steps_fast(5, 0.5, batch_size=batch_size)
    model.save_weights(this_weights)
    print("model saved: ", this_weights)

    score = run_validation(0.99)
    if score > best_score:
        print("Score improved from {} to {}. Switching to new best weights: {}".format(best_score, score, this_weights))
        best_score = score 
        best_weights = this_weights 
    else:
        print("Score didn't improve from previous best of {}. ".format(best_score))

    # epoch -> 25
    print("\n\n{} training block 3 {}".format('*'*17, '*'*17))
    this_weights = '../saved_models/{}/{}_{}_temp3.model_weights'.format(saved_folder, model_str, version)
    set_lr(model, 2e-5)
    for _ in range(5): 
        make_steps_fast(5, 0.25, batch_size=batch_size)
    model.save_weights(this_weights)
    print("model saved: ", this_weights)

    score = run_validation(0.99)
    if score > best_score:
        print("Score improved from {} to {}. Switching to new best weights: {}".format(best_score, score, this_weights))
        best_score = score 
        best_weights = this_weights 
    else:
        print("Score didn't improve from previous best of {}. ".format(best_score))


    # epoch -> 25
    print("\n\n{} training block 4 {}".format('*'*17, '*'*17))
    this_weights = '../saved_models/{}/{}_{}_temp4.model_weights'.format(saved_folder, model_str, version)
    set_lr(model, 2e-5)
    for _ in range(5): 
        make_steps_fast(5, 0.25, batch_size=batch_size)
    model.save_weights(this_weights)
    print("model saved: ", this_weights)

    score = run_validation(0.99)
    if score > best_score:
        print("Score improved from {} to {}. Switching to new best weights: {}".format(best_score, score, this_weights))
        best_score = score 
        best_weights = this_weights 
    else:
        print("Score didn't improve from previous best of {}. ".format(best_score))


    # epoch -> 25
    print("\n\n{} training block 5 {}".format('*'*17, '*'*17))
    this_weights = '../saved_models/{}/{}_{}_temp5.model_weights'.format(saved_folder, model_str, version)
    set_lr(model, 1e-5)
    for _ in range(5): 
        make_steps_fast(5, 0.1, batch_size=batch_size)
    model.save_weights(this_weights)
    print("model saved: ", this_weights)

    score = run_validation(0.99)
    if score > best_score:
        print("Score improved from {} to {}. Switching to new best weights: {}".format(best_score, score, this_weights))
        best_score = score 
        best_weights = this_weights 
    else:
        print("Score didn't improve from previous best of {}. ".format(best_score))


####### swich to 768 #######
img_shape = (768, 768, 3) 
batch_size = 20 

#print("\nSwitching to 768. Batch size: {}. Using best weights: {}".format(batch_size, best_weights))

model, branch_model, head_model = build_model(64e-5, 0, img_shape)

model.load_weights(best_weights)

run_validation(0.99)

print("\n{} training block 6 {}".format('*'*17, '*'*17))

branch_model.trainable = False
set_lr(model, 8e-5)
optim = Adam(lr=8e-5)
model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

make_steps_fast(10, 1, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp6_freezingBranch.model'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp6_freezingBranch.model'.format(saved_folder, model_str, version))

run_validation(0.99)

branch_model.trainable = True
set_lr(model, 8e-5)
optim = Adam(lr=8e-5)
model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

make_steps_fast(10, 1, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp6.model'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp6.model'.format(saved_folder, model_str, version))


set_lr(model, 8e-5)
print("\n{} training block 6.1 {}".format('*'*17, '*'*17))
make_steps_fast(20, 1, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp6.model'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp6.model'.format(saved_folder, model_str, version))

run_validation(0.99)

# epoch -> 25
set_lr(model, 4e-5)
print("\n\n{} training block 7 {}".format('*'*17, '*'*17))
for _ in range(4): 
    make_steps_fast(5, 0.5, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp7.model_weights'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp7.model_weights'.format(saved_folder, model_str, version))

run_validation(0.99)

# epoch -> 25
print("\n\n{} training block 8 {}".format('*'*17, '*'*17))
set_lr(model, 4e-5)
for _ in range(4): 
    make_steps_fast(5, 0.25, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp8.model_weights'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp8.model_weights'.format(saved_folder, model_str, version))

run_validation(0.99)

# epoch -> 25
print("\n\n{} training block 9 {}".format('*'*17, '*'*17))
set_lr(model, 2e-5)
for _ in range(4): 
    make_steps_fast(5, 0.25, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp9.model_weights'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp9.model_weights'.format(saved_folder, model_str, version))

run_validation(0.99)

# epoch -> 25
print("\n\n{} training block 10 {}".format('*'*17, '*'*17))
set_lr(model, 2e-5)
for _ in range(4): 
    make_steps_fast(5, 0.1, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp10.model_weights'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp10.model_weights'.format(saved_folder, model_str, version))

run_validation(0.99)

# epoch -> 25
print("\n\n{} training block 11 {}".format('*'*17, '*'*17))
set_lr(model, 2e-5)
for _ in range(4): 
    make_steps_fast(5, 0.05, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp11.model_weights'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp11.model_weights'.format(saved_folder, model_str, version))

run_validation(0.99)




####### swich to 1024 #######
prev_weights = '../saved_models/{}/{}_{}_temp11.model_weights'.format(saved_folder, model_str, version)

print("\n\n########### switching to 1024 ###########\n")
img_shape = (1024, 1024, 3) 
batch_size = 20 
thr = 0.99 

print("\nSwitching to 1024. Batch size: {}. Using best weights: {}".format(batch_size, prev_weights))
model, branch_model, head_model = build_model(64e-5, 0, img_shape)
model.load_weights(prev_weights)

model = multi_gpu_model(model, gpus=2)    
optim = Adam(lr=8e-5)
model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

run_validation(0.99)

################## freeze layers ##################
branch_model.trainable = False
set_lr(model, 8e-5)
optim = Adam(lr=8e-5)
model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

print("\n{} training block 1 {}".format('*'*17, '*'*17))
make_steps_fast(15, 1, batch_size=batch_size)

run_validation(thr)

model.save_weights('../saved_models/{}/{}_{}_temp12.model_weights'.format(saved_folder, model_str, version))
print("Weights saved: {}".format('../saved_models/{}/{}_{}_temp12.model_weights'.format(saved_folder, model_str, version)))
run_validation(0.99)


################## unfreeze layers ##################
branch_model.trainable = True
optim  = Adam(lr=8e-5)
model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

set_lr(model, 8e-5)
print("\n{} training block 1.1 {}".format('*'*17, '*'*17))
make_steps_fast(20, 1, batch_size=batch_size)

model.save_weights('../saved_models/{}/{}_{}_temp13.model_weights'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp113.model_weights'.format(saved_folder, model_str, version))
run_validation(thr)


print("\n{} training block 2 {}".format('*'*17, '*'*17))
set_lr(model, 8e-5)
for _ in range(4):
    make_steps_fast(5, 0.5, batch_size=batch_size)

model.save_weights('../saved_models/{}/{}_{}_temp14.model_weights'.format(saved_folder, model_str, version))
print("Weights saved: {}".format('../saved_models/{}/{}_{}_temp14.model_weights'.format(saved_folder, model_str, version)))
run_validation(thr)



print("\n\n{} training block 3 {}".format('*'*17, '*'*17))
set_lr(model, 4e-5)
for _ in range(4): 
    make_steps_fast(5, 0.25, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp15.model_weights'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp15.model_weights'.format(saved_folder, model_str, version))

run_validation(0.99)




print("\n\n{} training block 4 {}".format('*'*17, '*'*17))
set_lr(model, 2e-5)
for _ in range(4): 
    make_steps_fast(5, 0.25, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp16.model_weights'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp16.model_weights'.format(saved_folder, model_str, version))

run_validation(0.99)




# epoch -> 25
print("\n\n{} training block 5 {}".format('*'*17, '*'*17))
set_lr(model, 2e-5)
for _ in range(4): 
    make_steps_fast(5, 0.1, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp17.model_weights'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp17.model_weights'.format(saved_folder, model_str, version))

run_validation(0.99)



# epoch -> 25
print("\n\n{} training block 6 {}".format('*'*17, '*'*17))
set_lr(model, 2e-5)
for _ in range(4): 
    make_steps_fast(5, 0.05, batch_size=batch_size)
model.save_weights('../saved_models/{}/{}_{}_temp18.model_weights'.format(saved_folder, model_str, version))
print("model saved: ", '../saved_models/{}/{}_{}_temp18.model_weights'.format(saved_folder, model_str, version))

run_validation(0.99)




## TB final 
# final fine tuning - 30 x 5 epochs (can stop at any time)
print("\n{} training block final. Can stop any time! {}".format('*' * 17, '*' * 17))
print("\nStarted new round: lr: 1e-5, rounds: 2, epochs: 5, ampl: 0.25\n")
iters_to_run = 20
set_lr(model, 8e-6)
for iter1 in range(iters_to_run):
    if iter1 == 4:
        print("Setting LR to {}".format(4e-6))
        set_lr(model, 4e-6)
    if iter1 == 8: 
        print("Setting LR to {}".format(2e-6))
        set_lr(model, 2e-6)
    if iter1 == 12: 
        print("Setting LR to {}".format(1e-6))
        set_lr(model, 1e-6)

    ampl = 0.25 * ((iters_to_run - iter1) / iters_to_run)
    print('\n\nIter: {} ampl: {}'.format(iters_to_run, ampl))
    make_steps_fast(5, ampl, batch_size=batch_size)
    val_score = run_validation(0.99)
    model_path = '../saved_models/{}/{}_{}_final_{:.6f}.model_weights'.format(saved_folder, model_str, version, val_score)
    model.save_weights(model_path)
    print("Saved model:", model_path)

print("all done. ")











