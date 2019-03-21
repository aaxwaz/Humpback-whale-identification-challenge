
from a00_common_functions import *
from pandas import read_csv
import pickle
import numpy as np 
from os.path import isfile

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


def build_model_multich(channels, pretrained_weights, img_shape):

    model_multi_ch, branch_model_3ch, head_model_3ch = build_model(64e-5, 0, img_shape=img_shape)

    if pretrained_weights is not None:
        model, branch_model, head_model = build_model(64e-5, 0, img_shape=(img_shape[0], img_shape[1], 1))
        print("\nLoading weights from {} ".format(pretrained_weights))
        model.load_weights(pretrained_weights, by_name=False, skip_mismatch=False)

        print(branch_model.summary())
        for layer in branch_model_3ch.layers:
            print('Go for: {}'.format(layer.name))
            if layer.name == 'bm_conv2d_1':
                layer_old = branch_model.get_layer(layer.name)
                weights, bias = layer_old.get_weights()
                print('Old weights shape: {}'.format(weights.shape))
                print('Old bias shape: {}'.format(bias.shape))
                new_weights = np.zeros((9, 9, 3, 64), dtype=np.float32)
                new_weights[:, :, 0:0+1, :] = weights / channels
                new_weights[:, :, 1:1+1, :] = weights / channels
                new_weights[:, :, 2:2+1, :] = weights / channels
                layer.set_weights([new_weights, bias])
                continue
            layer_old = branch_model.get_layer(layer.name)
            weights = layer_old.get_weights()
            layer.set_weights(weights)
        for layer in head_model_3ch.layers:
            print('Go for: {}'.format(layer.name))
            layer_old = head_model.get_layer(layer.name)
            weights = layer_old.get_weights()
            layer.set_weights(weights)
        print(model_multi_ch.summary())
        
    return model_multi_ch, branch_model_3ch, head_model_3ch


def build_model_with_other_shape(pretrained_weights, img_shape=(512, 512, 1)):
    model, branch_model, head_model = build_model(64e-5, 0, img_shape=img_shape)
    print("\nLoading weights from {} ".format(pretrained_weights))
    model.load_weights(pretrained_weights, by_name=False, skip_mismatch=False)
    print(model.summary())
    return model, branch_model, head_model

def freeze_unfreeze_model(model, freeze=True, layer_name='bm_conv2d_1'):
    for layer in model.layers:
        if layer.name == layer_name:
            print("\nSetting layer {} trainable. \n".format(layer_name))
            layer.trainable = True
            continue
        if freeze:
            layer.trainable = False
        else:
            layer.trainable = True
    return model

def generate_train_val_4_fold_split(RUN_FOLD):

    def _find_train_val_split_4_fold(l, run_fold):
        assert len(l) > 1
        kf = KFold(n_splits=4)

        if len(l) == 2:
            return l, []
        elif len(l) == 3:
            if run_fold == 3:
                return l, []
            else:
                return l[:run_fold]+l[run_fold+1:], [l[run_fold]]
        else:
            tr, val = list(kf.split(l))[run_fold]
            return [l[i] for i in tr], [l[i] for i in val]

    def _expand_path_RGB(p):
        if isfile('../data/train_array_RGB/' + p): return '../data/train_array_RGB/' + p
        if isfile('../data/test_array_RGB/' + p): return '../data/test_array_RGB/' + p
        if isfile('../data/old/train_array/' + p): return '../data/old/train_array/' + p
        return None

    print("\n", "*"*70)
    print("Started generating train val splits ... \n")

    ### using zft bbox 
    temp_p2bb = pickle.load(open('../modified_data/p2bb_averaged_v1.pkl', 'rb'))
    p2bb1 = {}
    for k in temp_p2bb:
        p2bb1[k+'.npy'] = temp_p2bb[k]
    temp_p2bb = pickle.load(open('../modified_data/p2bb_averaged_playground_v1.pkl', 'rb'))
    p2bb2 = {}
    for k in temp_p2bb:
        p2bb2[k+'.npy'] = temp_p2bb[k]
    p2bb = {**p2bb1, **p2bb2}

    # load tagged data
    tagged = dict([(p[:-4]+'.npy',w) for _,p,w in read_csv('../data/train.csv').to_records()])
    newdata = dict([(k[:-4]+'.npy', v) for k,v in pickle.load(open('../modified_data/new_train_data.pkl', 'rb')).items()])
    newdata_bootstrap_Dec17 = dict([(k[:-4]+'.npy', v) for k,v in pickle.load(open('../modified_data/bootstrap_data/bootstrap_data_model_LB891.pkl', 'rb')).items()])
    tagged = dict(**tagged, **newdata)
    for ndk in newdata_bootstrap_Dec17:
        if ndk not in tagged:
            tagged[ndk] = newdata_bootstrap_Dec17[ndk]
        elif tagged[ndk] == newdata_bootstrap_Dec17[ndk]:
            if 0:
                print("Ignored duplicate keys with same values: {} -> {}".format(ndk, tagged[ndk]))
        else:
            print("Error. Found duplicate keys with different values: {} -> {}, {}".format(ndk, tagged[ndk], newdata_bootstrap_Dec17[ndk]))
            assert 0
    print("Total tagged images: ", len(tagged))

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
    submit = [f[:-4]+'.npy' for f in submit]
    join   = set(list(tagged.keys()) + submit)
    print(len(tagged),len(submit),len(join),list(tagged.items())[:5],submit[:5])

    # shuffle all data 
    np.random.seed(12321)
    for k in w2p:
        np.random.shuffle(w2p[k])

    valid_neg = []
    for k in w2p:
        if k == 'new_whale':
            valid_neg += w2p[k]
    valid_neg_batch = len(valid_neg) // 4
    print("Total new whale images: ", len(valid_neg))
    valid_neg = valid_neg[RUN_FOLD*valid_neg_batch:(RUN_FOLD+1)*valid_neg_batch][:1000]  # only select 1000 negative val images 
    print("Total neg images in val for this fold: ", len(valid_neg))

    train, val = [], []
    w2ts = {} # valid whale label to photos

    for k in w2p:
        if len(w2p[k]) > 1 and k != 'new_whale':
            train_split, val_split = _find_train_val_split_4_fold(w2p[k], RUN_FOLD)
            train += train_split
            val += val_split
            w2ts[k] = np.array(train_split).astype(object)

    train = sorted(train)
    val += valid_neg
    print("\nTotal train images: ", len(train))
    print("\nTotal val images: {}\n".format(len(val)))

    # photo -> image shape 
    p2size = {}
    for p in join:
        size      = np.load(_expand_path_RGB(p)).shape
        p2size[p] = size
    print(len(p2size), list(p2size.items())[:5])

    print("Done generating train val splits for fold {}! \n".format(RUN_FOLD))
    print("*"*70)

    return train, val, tagged, p2size
