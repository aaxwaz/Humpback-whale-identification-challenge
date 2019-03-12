# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import pickle
import os
import glob
import time
import cv2
import datetime
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold, train_test_split
from collections import Counter, defaultdict
import random
import shutil
import operator
import pyvips
from PIL import Image
import platform
import json
from tqdm import tqdm


if platform.processor() == 'Intel64 Family 6 Model 79 Stepping 1, GenuineIntel':
    DATASET_PATH = 'E:/Projects_M2/2018_07_Google_Open_Images/input/'
else:
    DATASET_PATH = 'D:/Projects/2018_07_Google_Open_Images/input/'

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
INPUT_PATH = ROOT_PATH + 'data/'
OUTPUT_PATH = ROOT_PATH + 'modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
FEATURES_PATH = ROOT_PATH + 'features/'
if not os.path.isdir(FEATURES_PATH):
    os.mkdir(FEATURES_PATH)
MODELS_PATH = ROOT_PATH + 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
CACHE_PATH = ROOT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)
HISTORY_FOLDER_PATH = MODELS_PATH + "history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)
SUBM_PATH = ROOT_PATH + 'subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)


UNIQUE_WHALES = 5005


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'), protocol=4)


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def show_image(im, name='image', type='rgb'):
    if type == 'bgr' or len(im.shape) == 2:
        cv2.imshow(name, im.astype(np.uint8))
    else:
        P1 = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow(name, P1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000, type='rgb'):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res, type=type)


def get_date_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def save_history_figure(history, path, columns=('fbeta', 'val_fbeta')):
    import matplotlib.pyplot as plt
    s = pd.DataFrame(history.history)
    plt.plot(s[list(columns)])
    plt.savefig(path)
    plt.close()


def read_single_image(path):
    try:
        img = pyvips.Image.new_from_file(path, access='sequential')
        img = np.ndarray(buffer=img.write_to_memory(),
                         dtype=np.uint8,
                         shape=[img.height, img.width, img.bands])
    except:
        print('Pyvips error! {}'.format(path))
        try:
            img = np.array(Image.open(path))
        except:
            try:
                img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            except:
                print('Fail')
                return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 2:
        img = img[:, :, :1]

    if img.shape[2] == 1:
        img = np.concatenate((img, img, img), axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img


def read_image_bgr_fast(path):
    img2 = read_single_image(path)
    img2 = img2[:, :, ::-1]
    return img2


def apk(actual, predicted, k=5):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=5):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def get_classes_array():
    train_df = pd.read_csv(INPUT_PATH + 'train.csv')
    unique_classes = sorted(train_df['Id'].unique())
    unique_classes.remove('new_whale')
    unique_classes = ['new_whale'] + unique_classes
    return unique_classes


def get_train_valid_data_for_training():
    cache_path = OUTPUT_PATH + 'train_valid_split.pklz'
    if not os.path.isfile(cache_path):
        CLASSES = get_classes_array()
        train_df = pd.read_csv(INPUT_PATH + 'train.csv')
        train = []
        valid = []
        for index, u in enumerate(CLASSES):
            part = train_df[train_df['Id'] == u].copy()
            if len(part) == 1:
                train.append((index, part['Image'].values[0]))
                # valid.append((index, part['Image'].values[0]))
            else:
                images = list(part['Image'].values)
                needed_images = 1
                if u == 'new_whale':
                    needed_images = 100
                for i in range(needed_images):
                    valid.append((index, images[i]))
                for i in range(needed_images, len(part)):
                    train.append((index, images[i]))
        print(len(train), len(valid))
        save_in_file((train, valid), cache_path)
    else:
        train, valid = load_from_file(cache_path)
    return train, valid


def compare_submissions(subm1, subm2):
    s1 = pd.read_csv(subm1).sort_values('Image')
    s2 = pd.read_csv(subm2).sort_values('Image')
    s1_keys = s1['Image'].values
    s2_keys = s2['Image'].values
    s1_word = s1['Id'].values
    s2_word = s2['Id'].values
    res = dict()
    res['exact_top1'] = 0
    res['exact_top2'] = 0
    res['exact_top3'] = 0
    res['exact_top4'] = 0
    res['exact_top5'] = 0
    for i in range(len(s1_keys)):
        if s1_keys[i] != s2_keys[i]:
            print('Check keys')
            exit()
        arr1 = s1_word[i].strip().split(' ')
        arr2 = s2_word[i].strip().split(' ')
        inter = 'intersection ' + str(len(set(arr1) & set(arr2)))
        if inter not in res:
            res[inter] = 0
        if arr1[0] == arr2[0]:
            res['exact_top1'] += 1
        if arr1[0] == arr2[0] and arr1[1] == arr2[1]:
            res['exact_top2'] += 1
        if arr1[0] == arr2[0] and arr1[1] == arr2[1] and arr1[2] == arr2[2]:
            res['exact_top3'] += 1
        if arr1[0] == arr2[0] and arr1[1] == arr2[1] and arr1[2] == arr2[2] and arr1[3] == arr2[3]:
            res['exact_top4'] += 1
        if tuple(arr1) == tuple(arr2):
            res['exact_top5'] += 1
        res[inter] += 1
    print('Compare {} and {}'.format(os.path.basename(subm1), os.path.basename(subm2)))
    for i in sorted(list(res.keys())):
        print('{} whales: {} ({:.2f}%)'.format(i, res[i], (100*res[i]) / len(s1_keys)))


def get_kfold_split_retinanet(nfolds):
    cache_path = OUTPUT_PATH + 'kfold_retinanet_{}.pklz'.format(nfolds)
    if not os.path.isfile(cache_path):
        train_df = pd.read_csv(INPUT_PATH + 'train.csv')
        unique_classes = sorted(train_df['Id'].unique())
        unique_classes.remove('new_whale')
        unique_classes = np.array(unique_classes)
        print(unique_classes.shape)

        # Stratified by whale IDs
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=66)
        ret = []
        for train_index, valid_index in kf.split(range(len(unique_classes))):
            train_classes = unique_classes[train_index]
            valid_classes = unique_classes[valid_index]
            train_images = list(train_df[train_df['Id'].isin(train_classes)]['Image'].values)
            valid_images = list(train_df[train_df['Id'].isin(valid_classes)]['Image'].values)
            ret.append([train_images.copy(), valid_images.copy()])

        # Append all new_whales
        new_whales = train_df[train_df['Id'] == 'new_whale']['Image'].values
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=66)
        index = 0
        for train_index, valid_index in kf.split(range(len(new_whales))):
            train_images = list(new_whales[train_index])
            valid_images = list(new_whales[valid_index])
            ret[index][0] += train_images
            ret[index][1] += valid_images
            print(len(set(ret[index][0])), len(set(ret[index][1])), len(set(ret[index][0])) + len(set(ret[index][1])))
            index += 1
        save_in_file(ret, cache_path)
    else:
        ret = load_from_file(cache_path)
    return ret


def check_submission_distribution(subm_path):
    s = pd.read_csv(subm_path)
    cl = dict()
    for index, row in s.iterrows():
        arr = row['Id'].strip().split(' ')
        for j in range(1):
            if arr[j] not in cl:
                cl[arr[j]] = 1
            else:
                cl[arr[j]] += 1

    res = sort_dict_by_values(cl)
    print(res[:20])
    print('Unique classes: {}'.format(len(res)))
    return res


def normalize_array(arr):
    arr = 255.0 * (arr - arr.min()) / (arr.max() - arr.min())
    return arr


def expand_path(p):
    if os.path.isfile(INPUT_PATH + 'train/' + p):
        return INPUT_PATH + 'train/' + p
    if os.path.isfile(INPUT_PATH + 'test/' + p):
        return INPUT_PATH + 'test/' + p
    if os.path.isfile(INPUT_PATH + 'playground/train/' + p):
        return INPUT_PATH + 'playground/train/' + p
    return p


def get_shape(f):
    img = pyvips.Image.new_from_file(f, access='sequential')
    shape = (img.height, img.width, img.bands)
    return shape