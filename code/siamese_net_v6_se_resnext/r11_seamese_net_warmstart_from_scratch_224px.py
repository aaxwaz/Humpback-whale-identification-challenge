# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

# Train single model just on playground data

if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

from a00_common_functions import *
from siamese_net_v6_se_resnext.a01_seamese_net_models_v6 import read_cropped_image, build_model, preprocess_image

from pandas import read_csv
from os.path import isfile
import pickle
import numpy as np
import os
from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.utils import Sequence
from lap import lapjv
import time
import math  
import pandas as pd 
from sklearn.model_selection import KFold
import keras
import random
from keras import backend as K
from scipy.ndimage import affine_transform
from multiprocessing.pool import ThreadPool
import operator
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {:.1f} s'.format(name, time.time() - t0))


########################################################
################# data preparation #####################
########################################################
BOX_SIZE = 224
LINEAR_ASSIGNMENT_SEGMENT_SIZE = 1
segment = True
if segment:
    print("Using LINEAR_ASSIGNMENT_SEGMENT_SIZE: {}".format(LINEAR_ASSIGNMENT_SEGMENT_SIZE))


def get_image_sizes(join):
    # photo -> image shape
    cache_path = CACHE_PATH + 'p2size_{}_warmstart.pklz'.format(os.path.basename(__file__))
    if not os.path.isfile(cache_path):
        p2size = {}
        for p in join:
            size = get_shape(expand_path(p))
            p2size[p] = size
        print(len(p2size), list(p2size.items())[:5])
        save_in_file(p2size, cache_path)
    else:
        p2size = load_from_file(cache_path)
        if len(p2size) != len(join):
            print('Len p2size diff!')
            exit()
    return p2size


def get_boxes():
    ### using zft bbox
    temp_p2bb = pickle.load(open(OUTPUT_PATH + 'p2bb_averaged_v1.pkl', 'rb'))
    p2bb1 = {}
    for k in temp_p2bb:
        p2bb1[k + '.jpg'] = temp_p2bb[k]
    temp_p2bb = pickle.load(open(OUTPUT_PATH + 'p2bb_averaged_playground_v1.pkl', 'rb'))
    p2bb2 = {}
    for k in temp_p2bb:
        p2bb2[k + '.jpg'] = temp_p2bb[k]
    p2bb = {**p2bb1, **p2bb2}
    return p2bb


def get_tagged_data():
    # load tagged data
    tagged = dict([(p[:-4] + '.jpg', w) for _, p, w in read_csv(INPUT_PATH + 'train.csv').to_records()])
    playground = dict([(p[:-4] + '.jpg', w) for _, p, w in read_csv(INPUT_PATH + 'playground/train.csv').to_records()])
    tagged = dict(**tagged, **playground)

    if 0:
        newdata = dict([(k[:-4] + '.jpg', v) for k, v in load_from_file_fast(OUTPUT_PATH + 'new_train_data.pkl').items()])
        newdata_bootstrap_Dec17 = dict([(k[:-4] + '.jpg', v) for k, v in load_from_file_fast(OUTPUT_PATH + 'bootstrap_data_model_LB891.pkl').items()])
        for ndk in newdata:
            if ndk not in tagged:
                tagged[ndk] = newdata[ndk]
        for ndk in newdata_bootstrap_Dec17:
            if ndk not in tagged:
                tagged[ndk] = newdata_bootstrap_Dec17[ndk]

    print("Total tagged images: ", len(tagged))
    return tagged


# whale ID -> photo
def get_whale_lists_for_class_dict(tagged):
    w2p = {}
    for img, lb in tagged.items():
        if img in p2bb:  # remove bad images!
            if lb not in w2p:
                w2p[lb] = [img]
            else:
                w2p[lb].append(img)
    return w2p


def print_stats(w2p):
    ### stats
    dfs = pd.DataFrame({'Id': list(w2p.keys())})
    dfs['counts'] = 0
    for i in range(dfs.shape[0]):
        dfs.iat[i, 1] = len(w2p[dfs.iat[i, 0]])
    print("\nStats: ")
    print('Num in class, Count')
    print(dfs['counts'].value_counts())


p2bb = get_boxes()
tagged = get_tagged_data()
w2p = get_whale_lists_for_class_dict(tagged)
print_stats(w2p)

# test
submit = [p for _, p, _ in read_csv(INPUT_PATH + 'sample_submission.csv').to_records()]
submit = [f[:-4] + '.jpg' for f in submit]
join = set(list(tagged.keys()) + submit)

# photo -> image shape
p2size = get_image_sizes(join)

print('Tagged items: {}'.format(len(tagged)))
print('Test items: {}'.format(len(submit)))
print('Overall items: {}'.format(len(join)))
print('Example: {} {}'.format(list(tagged.items())[:5], submit[:5]))

########################################################
################### CV preparation #####################
########################################################


def get_kfold_split_playground(w2p):
    # We use fold 0 for validation only
    fold_path = OUTPUT_PATH + 'kfold/new_4_folds_split_train_val_v2.pkl'
    data = load_from_file_fast(fold_path)
    data = data['fold_{}'.format(0)]

    train_part = []
    train = read_csv(INPUT_PATH + 'playground/train.csv')
    for index, row in train.iterrows():
        id = row['Image']
        clss = row['Id']
        if clss == 'new_whale':
            continue
        if len(w2p[clss]) < 2:
            continue
        train_part.append(id)

    valid_part = []
    for el in sorted(data['val']):
        valid_part.append(el[:-4] + '.jpg')

    new_train_part = []
    for el in sorted(data['train']):
        new_train_part.append(el[:-4] + '.jpg')

    return train_part, new_train_part, valid_part


def gen_train_labels_to_fotos(w2p, train):
    w2ts = {}  # valid whale label to photos
    for k in w2p:
        if len(w2p[k]) > 1 and k != 'new_whale':
            w2ts[k] = []
            for el in w2p[k]:
                if el in train:
                    w2ts[k].append(el)
            w2ts[k] = np.array(w2ts[k])
    return w2ts


train, new_train, val = get_kfold_split_playground(w2p)
w2ts = gen_train_labels_to_fotos(w2p, train)
print('Train size: {}'.format(len(train)))
print('Train size for validation: {}'.format(len(new_train)))
print('Valid size: {}'.format(len(val)))
print('W2TS size: {}'.format(len(w2ts)))


def read_for_training(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    x0, y0, x1, y1 = p2bb[p]
    img = read_cropped_image(expand_path(p), x0, y0, x1, y1, True, img_shape=(BOX_SIZE, BOX_SIZE, 3))
    # print('Train')
    # show_image(img)
    return img


def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    x0, y0, x1, y1 = p2bb[p]
    img = read_cropped_image(expand_path(p), x0, y0, x1, y1, False, img_shape=(BOX_SIZE, BOX_SIZE, 3))
    # print('Valid')
    # show_image(img)
    return img


def prepare_val_res(score, threshold):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    new_whale = 'new_whale'
    res = {}
    train_arr = np.array(new_train)

    for i, p in enumerate(val):
        t = []
        s = set()
        a = score[i, :]

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

        res[p[:-4] + '.jpg'] = t[:5]

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
        unroll_train_images = preprocess_image(unroll_train_images)

        ## a[0], b[0], a[1], b[1], a[2], b[2], a[3], a[4] ...
        a_indices = list(range(0, len(unroll_train_images), 2))
        a = unroll_train_images[a_indices]
        b_indices = list(range(1, len(unroll_train_images), 2))
        b = unroll_train_images[b_indices]
        c = np.zeros((size,1), dtype=K.floatx())
        for i in range(0, size, 2):
            c[i,   0] = 1 # This is a match
            c[i+1, 0] = 0 # Different whales

        return [a, b], c

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
    def __init__(self, data, batch_size=64, verbose=1):
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
        res = preprocess_image(res)

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
    batch = int(math.ceil(len(train) / LINEAR_ASSIGNMENT_SEGMENT_SIZE))
    for start in range(0, len(train), batch):
        end = min(len(train), start + batch)
        train_batch = train[start:end]

        features = branch_model.predict_generator(FeatureGen(train_batch, verbose=verbose), max_queue_size=12, verbose=0, use_multiprocessing=False)
        score    = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, verbose=0, use_multiprocessing=False)
        score    = score_reshape(score, features)
        res.append(score)

    return res


def make_steps_fast(step, ampl, verbose=2):
    """
    Perform training epochs
    @param step Number of epochs to perform`
    @param ampl the K, the randomized component of the score matrix.
    """
    global w2ts, t2i, steps, features, score, histories
    
    # shuffle the training pictures
    random.shuffle(train)
    
    # Map training picture hash value to index in 'train' array    
    t2i  = {}
    for i, t in enumerate(train): t2i[t] = i

    # Compute the match score for each picture pair of train data - square matrix score[i,j] is the score between image i and j, 0~1
    with timer("Compute score fast. "):
        score_list = compute_score_fast()    

    for i in range(len(score_list)):
        score_list[i] += ampl*np.random.random_sample(size=score_list[i].shape)
    
    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingDataFast(score_list, steps=step, batch_size=6),
        initial_epoch=steps, epochs=steps + step, max_queue_size=12, workers=6, verbose=verbose,
        #callbacks=[
        #    TQDMNotebookCallback(leave_inner=True, metric_format='{value:0.3f}')
        #]
        ).history
    steps += step
    
    # Collect history data
    history['epochs'] = steps
    history['ms'] = np.mean(score)
    history['lr'] = get_lr(model)
    print(history['epochs'], history['lr'], history['ms'])
    histories.append(history)


def run_validation(threshold):
    print("Running validation ...")
    print('Predict new train ({}) with branch model...'.format(len(new_train)))
    ftrain = branch_model.predict_generator(FeatureGen(new_train), max_queue_size=20, verbose=0, use_multiprocessing=False)
    print('Predict validation ({}) with branch model...'.format(len(val)))
    fval = branch_model.predict_generator(FeatureGen(val), max_queue_size=20, verbose=0, use_multiprocessing=False)
    print('Score table calc...')
    score = head_model.predict_generator(ScoreGen(ftrain, fval), max_queue_size=20, verbose=0, use_multiprocessing=False)
    score = score_reshape(score, ftrain, fval)
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

    score = np.mean(np.array(scores))
    print("\nValid score: {:.6f}".format(score))
    return score


########################################################
################# training starts ######################
########################################################

if __name__ == '__main__':
    histories = []
    score = 0
    steps = 0
    load_pretrained = True
    saved_folder = 'Res_v6_seresnext50'
    if not os.path.isdir(MODELS_PATH + '{}'.format(saved_folder)):
        os.makedirs(MODELS_PATH + '{}'.format(saved_folder))

    thr = 0.99
    version = 'warmstart'
    pretrained_weights = ''
    l2_rate = 0.0002
    model_str = 'seresnext50_224px'
    model, branch_model, head_model = build_model(4e-5, 0, img_shape=(BOX_SIZE, BOX_SIZE, 3))

    if load_pretrained and isfile(pretrained_weights):
        print("\nLoading weights from {} ".format(pretrained_weights))
        model.load_weights(pretrained_weights)

    ########### PLAN 6.1 procedures ###########
    # run_validation(thr)
    score = np.random.random_sample(size=(len(train), len(train)))

    branch_model.trainable = False
    set_lr(model, 2e-5)
    optim = Adam(lr=2e-5)
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

    print("\n{} training block 1 {}".format('*'*17, '*'*17))
    for i in range(4):
        make_steps_fast(5, 1000)
        score_valid = run_validation(thr)
        model.save_weights(MODELS_PATH + '{}/{}_{}_temp0_{}_{:.6f}.model'.format(saved_folder, model_str, version, i, score_valid))

    # 4 x 5 epochs - unfreeze layers
    branch_model.trainable = True
    set_lr(model, 2e-5)
    optim = Adam(lr=2e-5)
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

    for i in range(4):
        make_steps_fast(5, 1)
        score_valid = run_validation(thr)
        model.save_weights(MODELS_PATH + '{}/{}_{}_temp04_{}_{:.6f}.model'.format(saved_folder, model_str, version, i, score_valid))

    print("\n{} training block 2 {}".format('*'*17, '*'*17))
    set_lr(model, 1e-5)
    for _ in range(4):
        make_steps_fast(5, 0.5)
    score_valid = run_validation(thr)
    model.save_weights(MODELS_PATH + '{}/{}_{}_temp1_{:.6f}.model'.format(saved_folder, model_str, version, score_valid))

    # 4 x 5 epochs
    print("\n{} training block 3 {}".format('*'*17, '*'*17))
    set_lr(model, 1e-5)
    for _ in range(4):
        make_steps_fast(5, 0.25)
        score_valid = run_validation(thr)
        model.save_weights(MODELS_PATH + '{}/{}_{}_temp2_{:.6f}.model'.format(saved_folder, model_str, version, score_valid))

    # continue fine tuning - 4 x 5 epochs
    print("\n{} training block 4 {}".format('*'*17, '*'*17))
    print("\n\nStarted new round: lr: 2e-5, rounds: 5, epochs: 5, ampl: 0.25\n")
    set_lr(model, 1e-5)
    for _ in range(4):
        make_steps_fast(5, 0.25)
        score_valid = run_validation(thr)
        save_path = MODELS_PATH + '{}/{}_{}_temp3_{:.6f}.model'.format(saved_folder, model_str, version, score_valid)
        model.save_weights(save_path)
        print("Saved model: ", save_path)

    # final fine tuning - 4 x 5 epochs
    print("\n{} training block 5 {}".format('*'*17, '*'*17))
    print("\nStarted new round: lr: 1e-5, rounds: 2, epochs: 5, ampl: 0.25\n")
    set_lr(model, 1e-5)
    for _ in range(4):
        make_steps_fast(5, 0.25)
        score_valid = run_validation(thr)
        save_path = MODELS_PATH + '{}/{}_{}_temp4_{:.6f}.model'.format(saved_folder, model_str, version, score_valid)
        model.save_weights(save_path)
        print("Saved model: ", save_path)

    # final fine tuning - 20 x 5 epochs
    print("\n{} training block 6 {}".format('*' * 17, '*' * 17))
    print("\nStarted new round: lr: 1e-5, rounds: 2, epochs: 5, ampl: 0.25\n")
    set_lr(model, 8e-6)
    for _ in range(20):
        make_steps_fast(5, 0.25)
        val_score = run_validation(thr)
        model_path = MODELS_PATH + '{}/{}_{}_final_{:.6f}.model'.format(saved_folder, model_str, version, val_score)
        model.save_weights(model_path)
        print("Saved model:", model_path)

    print("all done. ")
    #################################


'''
TB 1: Valid score: 0.773800
TB 2: Valid score: 0.849076
TB 4: Valid score: 0.860118
'''
