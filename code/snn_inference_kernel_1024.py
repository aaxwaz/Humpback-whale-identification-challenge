from pandas import read_csv
from os.path import isfile
from PIL import Image as pil_image
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
#import hungarian
import operator
# Suppress annoying stderr output when importing keras.
import sys
import platform
old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')
import keras
sys.stderr = old_stderr
from albumentations import * 
import cv2
import random
from keras import backend as K
from keras.preprocessing.image import img_to_array,array_to_img
from scipy.ndimage import affine_transform
from multiprocessing.pool import ThreadPool

from a02_model_modifications import * 
from keras.utils import multi_gpu_model

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"    
print("\n\nUsing GPU: {}\n\n".format(os.environ["CUDA_VISIBLE_DEVICES"])) 

anisotropy   = 2.15 # The horizontal compression ratio
crop_margin  = 0.05 # The margin added around the bounding box to compensate for bounding box inaccuracy
rotate = {}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_weights_1", default="", help="best trained weights for model 1", type=str)
parser.add_argument("--model_weights_2", default="", help="best trained weights for model 2", type=str)
parser.add_argument("--model_weights_3", default="", help="best trained weights for model 3", type=str)
parser.add_argument("--model_weights_4", default="", help="best trained weights for model 4", type=str)
args = parser.parse_args()


########################################################
################# data preparation #####################
########################################################

# Read the dataset description
# train 
tagged = dict([(p[:-4]+'.jpg',w) for _,p,w in read_csv('../data/train.csv').to_records()])

# test
submit = [p for _,p,_ in read_csv('../data/sample_submission.csv').to_records()]
submit = [f[:-4]+'.jpg' for f in submit]
join   = list(tagged.keys()) + submit
print(len(tagged),len(submit),len(join),list(tagged.items())[:5],submit[:5])

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

# find out all excluded images 
exclude = []
for img in join:
    if img not in p2bb:
        exclude.append(img)
print("Total excluded: ", len(exclude))

def expand_path_RGB(p):
    if isfile('../data/train/' + p): return '../data/train/' + p
    if isfile('../data/test/' + p): return '../data/test/' + p
    #if isfile('../data/train_array_RGB/' + p): return '../data/train_array_RGB/' + p
    #if isfile('../data/test_array_RGB/' + p): return '../data/test_array_RGB/' + p
    #if isfile('../data/old/train_array/' + p): return '../data/old/train_array/' + p
    return None


########################################################
############## Siamese Neural Networks #################
########################################################


def subblock(x, filter, block_number, **kwargs):
    from keras import backend as K
    from keras.layers import Activation, Add, BatchNormalization, Conv2D

    x = BatchNormalization(name='bm_bn_1_sub_{}'.format(block_number))(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', name='bm_conv2d_1_sub_{}'.format(block_number), **kwargs)(y)  # Reduce the number of features to 'filter'
    y = BatchNormalization(name='bm_bn_2_sub_{}'.format(block_number))(y)
    y = Conv2D(filter, (3, 3), activation='relu', name='bm_conv2d_2_sub_{}'.format(block_number), **kwargs)(y)  # Extend the feature field
    y = BatchNormalization(name='bm_bn_3_sub_{}'.format(block_number))(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), name='bm_conv2d_3_sub_{}'.format(block_number), **kwargs)(y)  # no activation # Restore the number of original features
    y = Add(name='bm_add_{}'.format(block_number))([x, y])  # Add the bypass connection
    y = Activation('relu', name='bm_relu_sub_{}'.format(block_number))(y)
    return y

def build_model(lr, l2, img_shape=(384, 384, 1), activation='sigmoid'):
    from keras import backend as K
    from keras import regularizers
    from keras.optimizers import Adam
    from keras.engine.topology import Input
    from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, \
        GlobalMaxPooling2D, \
        Lambda, MaxPooling2D, Reshape
    from keras.models import Model, load_model

    from keras.layers import GlobalAveragePooling2D, multiply, Permute

    ##############
    # BRANCH MODEL
    ##############
    regul = regularizers.l2(l2)
    optim = Adam(lr=lr)
    kwargs = {'padding': 'same', 'kernel_regularizer': regul}

    inp = Input(shape=img_shape, name='bm_input_1')  # 384x384x1
    x = Conv2D(64, (9, 9), strides=2, activation='relu', name='bm_conv2d_1', **kwargs)(inp)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='bm_mp_2')(x)  # 96x96x64
    for _ in range(2):
        x = BatchNormalization(name='bm_bn_2_{}'.format(_))(x)
        x = Conv2D(64, (3, 3), activation='relu', name='bm_relu_2_{}'.format(_), **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='bm_mp_3')(x)  # 48x48x64
    x = BatchNormalization(name='bm_bn_3')(x)
    x = Conv2D(128, (1, 1), activation='relu', name='bm_conv2d_3', **kwargs)(x)  # 48x48x128
    for _ in range(4):
        x = subblock(x, 64, 0 + _, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='bm_mp_4')(x)  # 24x24x128
    x = BatchNormalization(name='bm_bn_4')(x)
    x = Conv2D(256, (1, 1), activation='relu', name='bm_conv2d_4', **kwargs)(x)  # 24x24x256
    for _ in range(4):
        x = subblock(x, 64, 4 + _, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='bm_mp_5')(x)  # 12x12x256
    x = BatchNormalization(name='bm_bn_5')(x)
    x = Conv2D(384, (1, 1), activation='relu', name='bm_conv2d_5', **kwargs)(x)  # 12x12x384
    for _ in range(4):
        x = subblock(x, 96, 8 + _, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='bm_mp_6')(x)  # 6x6x384
    x = BatchNormalization(name='bm_bn_6')(x)
    x = Conv2D(512, (1, 1), activation='relu', name='bm_conv2d_6', **kwargs)(x)  # 6x6x512
    for _ in range(4):
        x = subblock(x, 128, 12 + _, **kwargs)

    x = GlobalMaxPooling2D(name='gmp_2d')(x)  # 512
    branch_model = Model(inp, x, name='branch_model')

    ############
    # HEAD MODEL
    ############
    mid = 32
    xa_inp = Input(shape=branch_model.output_shape[1:], name='hm_inp_a')
    xb_inp = Input(shape=branch_model.output_shape[1:], name='hm_inp_b')
    x1 = Lambda(lambda x: x[0] * x[1], name='lambda_1')([xa_inp, xb_inp])
    x2 = Lambda(lambda x: x[0] + x[1], name='lambda_2')([xa_inp, xb_inp])
    x3 = Lambda(lambda x: K.abs(x[0] - x[1]), name='lambda_3')([xa_inp, xb_inp])
    x4 = Lambda(lambda x: K.square(x), name='lambda_4')(x3)
    x = Concatenate(name='concat_1')([x1, x2, x3, x4])
    x = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x = Conv2D(mid, (4, 1), activation='relu', padding='valid', name='hm_conv_2d_1')(x)
    x = Reshape((branch_model.output_shape[1], mid, 1), name='hm_reshape_2')(x)
    x = Conv2D(1, (1, mid), activation='linear', padding='valid', name='hm_conv_2d_2')(x)
    x = Flatten(name='flatten')(x)

    # Weighted sum implemented as a Dense layer.
    x = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head_model')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a = Input(shape=img_shape)
    img_b = Input(shape=img_shape)
    xa = branch_model(img_a)
    xb = branch_model(img_b)
    x = head_model([xa, xb])
    model = Model([img_a, img_b], x, name='full_model')
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model

########################################################
################# training preparation #################
########################################################


def read_raw_image_npy(p):
    img = img_to_array(pil_image.open(expand_path_RGB(p))).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[-1] == 3:
        return img 
    elif len(img.shape) == 2:
        return np.stack((img,)*3, axis=-1)
    else:
        return np.stack((img[:,:,0],)*3, axis=-1)


def stronger_aug_weimin(p=.5):
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
        augm = stronger_aug_weimin(p=1.0)(image=img, bboxes=bb, labels=['labels'])
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

def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image_npy(p, False)


# Find all the whales associated with an image id. It can be ambiguous as duplicate images may have different whale ids.
#h2ws = {}  # hash to whale id 
#new_whale = 'new_whale'
#for p,w in tagged.items():
#    if w != new_whale: # Use only identified whales
#        h = p2h[p]
#        if h not in h2ws: h2ws[h] = []
#        if w not in h2ws[h]: h2ws[h].append(w)
#for h,ws in h2ws.items():
#    if len(ws) > 1:
#        h2ws[h] = sorted(ws)
#len(h2ws)



# Find the list of training images, keep only whales with at least two images.
#train = [] # A list of training image ids
#for hs in w2hs.values():
#    if len(hs) > 1:
#        train += hs
#random.shuffle(train)
#train_set = set(train)

#w2ts = {} # Associate the image ids from train to each whale id - whale id to hashes in np array 
#for w,hs in w2hs.items():
#    for h in hs:
#        if h in train_set:
#            if w not in w2ts: w2ts[w] = []
#            if h not in w2ts[w]: w2ts[w].append(h)
#for w,ts in w2ts.items(): w2ts[w] = np.array(ts)
    
#t2i = {} # The position in train of each training image id
#for i,t in enumerate(train): t2i[t] = i

#len(train),len(w2ts)


# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, batch_size=24, verbose=1):
        super(FeatureGen, self).__init__()
        self.data       = data
        self.batch_size = batch_size
        self.verbose    = verbose
        #if self.verbose > 0: self.progress = tqdm_notebook(total=len(self), desc='Features')
        #if self.verbose > 0: self.progress = range(total=len(self))

    def __getitem__(self, index):
        start = self.batch_size*index
        size  = min(len(self.data) - start, self.batch_size)
        a     = np.zeros((size,) + img_shape, dtype=K.floatx())
        for i in range(size): 
            a[i,:,:,:] = read_for_validation(self.data[start + i])
        #if self.verbose > 0: 
        #    self.progress.update()
        #    if self.progress.n >= len(self): self.progress.close()
        return a

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


########################################################
################# inference part #######################
########################################################

# Not computing the submission in this notebook because it is a little slow. It takes about 15 minutes on setup with a GTX 1080.


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x
   
def prepare_submission2_fast(score, threshold, filename, most_common = ['new_whale'] * 5):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    new_whale = 'new_whale'

    with gzip.open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i,p in enumerate(submit):
            t = []
            s = set()
            a = score[i,:]

            top_label_probs = {}
            cond = a > threshold
            cond_index = np.where(cond)[0]
            cond_images = known[cond_index]
            for j, img in enumerate(cond_images):
                if tagged[img] in top_label_probs:
                    top_label_probs[tagged[img]] += a[cond_index[j]]
                else:
                    top_label_probs[tagged[img]] = a[cond_index[j]]

            sorted_top_label_probs = sort_dict_by_values(top_label_probs)
            #print(sorted_top_label_probs, "\n")
            #if i == 3:
            #    print(cond_index, cond_images, top_label_probs)

            t = []
            for lb, _ in sorted_top_label_probs:
                t.append(lb)

            if len(t) < 5:
                t.append(new_whale)

            for index in np.argsort(a)[::-1]:
                if tagged[known[index]] not in t:
                    t.append(tagged[known[index]])
                if len(t) >= 5:
                    break

            assert len(t) >= 5

            f.write(p[:-4]+'.jpg' + ',' + ' '.join(t[:5]) + '\n')
        if len(submit_excluded) > 0:
            for e in submit_excluded:
                f.write(e[:-4]+'.jpg' + ',' + ' '.join(most_common[:5]) + '\n')


def prepare_submission3_average(score, threshold, filename, most_common = ['new_whale'] * 5):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    new_whale = 'new_whale'

    with gzip.open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i,p in enumerate(submit):
            t = []
            s = set()
            a = score[i,:]

            top_label_probs = {}
            cond = a > threshold
            cond_index = np.where(cond)[0]
            cond_images = known[cond_index]
            for j, img in enumerate(cond_images):
                if tagged[img] in top_label_probs:
                    top_label_probs[tagged[img]] += [a[cond_index[j]]]
                else:
                    top_label_probs[tagged[img]] = [a[cond_index[j]]]
            print("\n\n",top_label_probs )
            for k in top_label_probs:
                top_label_probs[k] = np.mean(top_label_probs[k])
            print(top_label_probs )

            sorted_top_label_probs = sort_dict_by_values(top_label_probs)
            #print(sorted_top_label_probs, "\n")
            #if i == 3:
            #    print(cond_index, cond_images, top_label_probs)

            t = []
            for lb, _ in sorted_top_label_probs:
                t.append(lb)

            if len(t) < 5:
                t.append(new_whale)

            for index in np.argsort(a)[::-1]:
                if tagged[known[index]] not in t:
                    t.append(tagged[known[index]])
                if len(t) >= 5:
                    break

            assert len(t) >= 5

            f.write(p[:-4]+'.jpg' + ',' + ' '.join(t[:5]) + '\n')
        if len(submit_excluded) > 0:
            for e in submit_excluded:
                f.write(e[:-4]+'.jpg' + ',' + ' '.join(most_common[:5]) + '\n')

def check_num_newwhales(score, thre):
    score_mask = score < thre
    print("New whales: ", np.sum(np.all(score_mask, axis=1)))


def prepare_val_res(train, val, score, threshold):
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

# Find elements from training sets not 'new_whale'
#h2ws = {}
known = []
for p,w in tagged.items():
    if w != 'new_whale': # Use only identified whales
        if p in exclude:
            print("Excluding image {} from training set. ".format(p))
        else:
            known.append(p)
known = sorted(known)
known = np.array(known)
print("Total known images: ", len(known))
print("known images: ", known[:10])

submit_excluded = []
for i in submit:
    if i in exclude:
        submit_excluded.append(i)
submit = [i for i in submit if i not in submit_excluded]
print("Total excluded submit: ", len(submit_excluded))

# Dictionary of picture indices
#h2i   = {}
#for i,h in enumerate(known): h2i[h] = i

## make predictions 
## fine tune model (LB - 0.87)


img_shape = (1024, 1024, 3)

if 1:
    model1_score = '../modified_data/saved_scores/PLAN21_martin_diffAlbu_384_to_1024_fold0.pkl'
    model1_weights = args.model_weights_1

    if isfile(model1_score):
        score1 = pickle.load(open(model1_score, 'rb'))
        print("Score1 loaded.")
    else:
        print("\n\nModel 1")
        pretrain_weights = model1_weights
        K.clear_session()
        model, branch_model, head_model = build_model(64e-5, 0, img_shape)
        print("\nLoading model from {} ".format(pretrain_weights))

        model = multi_gpu_model(model, gpus=2)    
        optim = Adam(lr=8e-5)
        model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

        model.load_weights(pretrain_weights)
        fknown  = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
        fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
        score1   = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
        score1   = score_reshape(score1, fknown, fsubmit)
        with open(model1_score, 'wb') as f:
            pickle.dump(score1, f)

if 1:
    model2_score = '../modified_data/saved_scores/PLAN21_martin_diffAlbu_384_to_1024_fold1.pkl'

    #model2_weights = '../saved_models/PLAN12/mpiotte-standard-PLAN12_withThreeChannel_Fold_1-final.model_weights'
    model2_weights =  args.model_weights_2

    if isfile(model2_score):
        score2 = pickle.load(open(model2_score, 'rb'))
        print("Score2 loaded.")
    else:
        print("\n\nModel 2")
        pretrain_weights = model2_weights
        K.clear_session()
        model, branch_model, head_model = build_model(64e-5, 0, img_shape)
        print("\nLoading model from {} ".format(pretrain_weights))

        model = multi_gpu_model(model, gpus=2)    
        optim = Adam(lr=8e-5)
        model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

        model.load_weights(pretrain_weights)
        fknown  = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
        fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
        score2   = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
        score2   = score_reshape(score2, fknown, fsubmit)
        with open(model2_score, 'wb') as f:
            pickle.dump(score2, f)


if 1:
    model3_score = '../modified_data/saved_scores/PLAN21_martin_diffAlbu_384_to_1024_fold2.pkl'
    model3_weights = args.model_weights_3

    if isfile(model3_score):
        score3 = pickle.load(open(model3_score, 'rb'))
        print("Score3 loaded.")
    else:
        print("\n\nModel 3")
        pretrain_weights = model3_weights
        K.clear_session()
        model, branch_model, head_model = build_model(64e-5, 0, img_shape)
        print("\nLoading model from {} ".format(pretrain_weights))

        model = multi_gpu_model(model, gpus=2)    
        optim = Adam(lr=8e-5)
        model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

        model.load_weights(pretrain_weights)
        fknown  = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
        fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
        score3   = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
        score3   = score_reshape(score3, fknown, fsubmit)
        with open(model3_score, 'wb') as f:
            pickle.dump(score3, f)


if 1:
    model4_score = '../modified_data/saved_scores/PLAN21_martin_diffAlbu_384_to_1024_fold3.pkl'
    model4_weights = args.model_weights_4
    #model4_weights = '../saved_models/PLAN12/mpiotte-standard-PLAN12_withThreeChannel_Fold_3-temp5.model_weights'

    if isfile(model4_score):
        score4 = pickle.load(open(model4_score, 'rb'))
        print("Score4 loaded.")
    else:
        print("\n\nModel 4")
        pretrain_weights = model4_weights
        K.clear_session()
        model, branch_model, head_model = build_model(64e-5, 0, img_shape)
        print("\nLoading model from {} ".format(pretrain_weights))

        model = multi_gpu_model(model, gpus=2)    
        optim = Adam(lr=8e-5)
        model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

        model.load_weights(pretrain_weights)
        fknown  = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
        fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
        score4   = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
        score4   = score_reshape(score4, fknown, fsubmit)
        with open(model4_score, 'wb') as f:
            pickle.dump(score4, f)


if 1:
    score_model_PLAN21_martin_1024_diffAlbu = (score1 + score2 + score3 + score4) / 4
    res = {}

    EPS = 0.0001 
    score_model_PLAN21_martin_1024_diffAlbu[score_model_PLAN21_martin_1024_diffAlbu<EPS] = 0
    from scipy import sparse 
    score_model_PLAN21_martin_1024_diffAlbu = sparse.csr_matrix(score_model_PLAN21_martin_1024_diffAlbu)

    res['score'] = score_model_PLAN21_martin_1024_diffAlbu
    res['row_names'] = submit
    res['col_names'] = known 

    if not os.path.isdir('../features/'):
        os.mkdir('../features/')

    pickle.dump(res, open('../features/Res_kernel_1024_averged_score_matrix_res_test_vs_train.pkl', 'wb'))

    print("Done. Score saved. ")


