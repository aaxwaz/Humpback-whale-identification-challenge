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
import operator
# Suppress annoying stderr output when importing keras.
import sys
import platform
old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')
import keras
sys.stderr = old_stderr
import shutil
import random
from keras import backend as K
from keras.preprocessing.image import img_to_array,array_to_img
from scipy.ndimage import affine_transform
from multiprocessing.pool import ThreadPool
from scipy.stats.mstats import gmean

from a02_model_modifications import * 
from scipy.stats.mstats import gmean

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

# find out all excluded images - will simply give labels of ['new_whale'] * 5 in final submit
exclude = []
for img in join:
    if img not in p2bb:
        exclude.append(img)
print("Total excluded: ", len(exclude))

# Find elements from training sets not 'new_whale'
known = []
for p,w in tagged.items():
    if w != 'new_whale': # Use only identified whales
        if p in exclude:
            print("Excluding image {} from training set. ".format(p))
        else:
            known.append(p)
known = sorted(known)
known = np.array(known)
print("\nTotal known images: ", len(known))
#print("known images: ", known[:10])

submit_excluded = []
for i in submit:
    if i in exclude:
        submit_excluded.append(i)
submit = [i for i in submit if i not in submit_excluded]
print("Total excluded submit: ", len(submit_excluded))
print(submit_excluded)

submit_array = np.array(submit)


########################################################
################# util functions #######################
########################################################

def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x
   
def get_overwrite_res(score, submit, known, min_thre, top_n=1):
    """Generating overwriting predictions based on train vs test ranking"""

    res = {}
    for i, k in enumerate(known):
        temp_score_index = np.argsort(score[:, i])[-top_n:][::-1]
        for ind in temp_score_index:
            if score[:, i][ind] < min_thre:
                break
            this_score = score[:, i][ind]
            this_test_image = submit[ind]
            if this_test_image in res:
                if tagged[k] != res[this_test_image]['label']:
                    if this_score > res[this_test_image]['score']:
                        res[this_test_image]['label'] = tagged[k]
                        res[this_test_image]['score'] = this_score
            else:
                res[this_test_image] = {}
                res[this_test_image]['score'] = this_score
                res[this_test_image]['label'] = tagged[k]

    return res

def prepare_val_res(train, val, score, threshold):

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

###############################################################
################# Ensembling preds and post proc ##############
###############################################################

if 1: # kernel model -> 384 - 1024,

    print("Starting for kernel martin with 1024 ")

    # score 1
    dat = pickle.load(open('../features/Res_kernel_1024_averged_score_matrix_res_test_vs_train.pkl', 'rb'))
    te_name = np.array(dat['row_names'])
    tr_name = np.array(dat['col_names'])
    kernelModel     = dat['score']
    te_name = np.array([i[:-4]+'.jpg' for i in te_name])
    tr_name = np.array([i[:-4]+'.jpg' for i in tr_name])

    ## test rows 
    xsorted = np.argsort(te_name)
    ypos    = np.searchsorted(te_name[xsorted], submit_array)
    indices = xsorted[ypos]
    kernelModel = kernelModel[indices]
    assert np.all(te_name[indices] == submit_array)
    ## train cols 
    xsorted = np.argsort(tr_name)
    ypos    = np.searchsorted(tr_name[xsorted], known)
    indices = xsorted[ypos]
    kernelModel = kernelModel[:,indices]
    assert np.all(tr_name[indices] == known)


if 1: # 959 densenet model 512
    print("Starting for 959 densenet 512 model ...")

    # score 1
    dat = pickle.load(open('../features/cv-analysis-fs14-LB959-densenet121-512px-sparse-test.pkl', 'rb'))
    te_name = np.array(dat['row_names'])
    tr_name = np.array(dat['col_names'])
    densenet     = dat['test_vs_train_mat_sparse']

    ## test rows 
    xsorted = np.argsort(te_name)
    ypos    = np.searchsorted(te_name[xsorted], submit_array)
    indices = xsorted[ypos]
    densenet = densenet[indices]
    assert np.all(te_name[indices] == submit_array)
    ## train cols 
    xsorted = np.argsort(tr_name)
    ypos    = np.searchsorted(tr_name[xsorted], known)
    indices = xsorted[ypos]
    densenet = densenet[:,indices]
    assert np.all(tr_name[indices] == known)


if 1: # 959 SERESNEXT model 
    print("Starting for 959 SE-ResNeXt model ...")

    # score 1
    dat = pickle.load(open('../features/cv-analysis-fs16-LB959-seresnext50-384px-sparse-test.pkl', 'rb'))
    te_name = np.array(dat['row_names'])
    tr_name = np.array(dat['col_names'])
    seresnext     = dat['test_vs_train_mat_sparse']

    ## test rows 
    xsorted = np.argsort(te_name)
    ypos    = np.searchsorted(te_name[xsorted], submit_array)
    indices = xsorted[ypos]
    seresnext = seresnext[indices]
    assert np.all(te_name[indices] == submit_array)

    ## train cols 
    xsorted = np.argsort(tr_name)
    ypos    = np.searchsorted(tr_name[xsorted], known)
    indices = xsorted[ypos]
    seresnext = seresnext[:,indices]
    assert np.all(tr_name[indices] == known)


###########################################
# submit 
###########################################

late_submit = (densenet + seresnext)*0.6*0.5 + kernelModel*0.4

final_score_with_stacking = late_submit.toarray()

###########################################
# post processing  
###########################################

res = prepare_val_res(known, submit_array, final_score_with_stacking, 0.7) 

min_thre_for_train_vs_test = 0.46

all_classes = []
for k in tagged:
    if tagged[k] not in all_classes:
        all_classes.append(tagged[k])
class_used_top1 = {}
class_used_top2 = {}

for k in all_classes:
    class_used_top1[k] = 0 
    class_used_top2[k] = 0 

for k in res:
    top_class = res[k][0] 
    if top_class in class_used_top1:
        class_used_top1[top_class] += 1 
    else:
        class_used_top1[top_class] = 1

for k in res:
    top_class = res[k][1] 
    if top_class in class_used_top2:
        class_used_top2[top_class] += 1 
    else:
        class_used_top2[top_class] = 1

total_new = 0 
for k in res:
    if res[k][0] == 'new_whale':
        total_new += 1
print("Total new before post processing: ", total_new)

flip_res = get_overwrite_res(final_score_with_stacking, submit_array, known, min_thre=min_thre_for_train_vs_test, top_n=1)

diff = {}
for k in flip_res:
    if flip_res[k]['label'] != res[k][0]:
        diff[k] = {}
        diff[k]['flip_label'] = flip_res[k]['label']
        diff[k]['old_pred'] = res[k]
        diff[k]['flip_score'] = flip_res[k]['score']
print("Total {} cases where first prediction is not flip_res".format(len(diff)))

records = []

### start post processing based on train vs test ranking 
total_changed = 0 

for k in diff:
    if diff[k]['flip_label'] == res[k][1]:
        if res[k][0] == 'new_whale':

            if class_used_top1[diff[k]['flip_label']] > 0:
                continue 
            if class_used_top2[diff[k]['flip_label']] == 3 and diff[k]['flip_score'] < 0.6: 
                continue 

            total_changed += 1
            temp = res[k][0]
            res[k][0] = diff[k]['flip_label']
            res[k][1] = temp

        else:
            total_changed += 1
            temp = res[k][0]
            res[k][0] = diff[k]['flip_label']
            res[k][1] = temp

### final stats and submit!
total_new = 0 
for k in res:
    if res[k][0] == 'new_whale':
        total_new += 1
print("Total new after post processing: ", total_new)

if not os.path.isdir('../submission'):
    os.mkdir('../submission')
    
filename = '../submission/final_submit_with_post_proc.csv'   
with open(filename, 'wt', newline='\n') as f:
    f.write('Image,Id\n')
    for p in submit:
        f.write(p[:-4]+'.jpg' + ',' + ' '.join(res[p]) + '\n')
    for exc in submit_excluded:
        f.write(exc + ',' + ' '.join(['new_whale']*5) + '\n') 