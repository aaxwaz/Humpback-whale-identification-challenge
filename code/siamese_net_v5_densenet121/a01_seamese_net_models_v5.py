# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
from albumentations import *


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


def get_branch_model_v1(inp_shape):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    model = InceptionResNetV2(input_shape=inp_shape, include_top=False, weights='imagenet', pooling='max')
    return model


def get_branch_model(inp_shape):
    from keras.applications.densenet import DenseNet121
    model = DenseNet121(input_shape=inp_shape, include_top=False, weights='imagenet', pooling='max')
    return model


def build_model(lr, l2, activation='sigmoid', img_shape=(224, 224, 3)):
    from keras import backend as K
    from keras import regularizers
    from keras.optimizers import Adam
    from keras.engine.topology import Input
    from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, \
        GlobalMaxPooling2D, \
        Lambda, MaxPooling2D, Reshape
    from keras.models import Model, load_model

    optim = Adam(lr=lr)

    ##############
    # BRANCH MODEL
    ##############

    branch_model = get_branch_model(img_shape)

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
    head_model = Model([xa_inp, xb_inp], x, name='head')

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


def strong_aug(p=.5):
    return Compose([
        HorizontalFlip(p=0.001),
        VerticalFlip(p=0.001),
        RandomRotate90(p=0.001),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.1),
        OneOf([
            MotionBlur(p=0.5),
            MedianBlur(blur_limit=3, p=0.5),
            Blur(blur_limit=3, p=0.5),
        ], p=0.1),
        ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=20, p=0.9),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
        ], p=0.1),
        OneOf([
            RGBShift(p=1.0, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
            HueSaturationValue(p=1.0),
        ], p=0.5),
        JpegCompression(p=0.2, quality_lower=55, quality_upper=99),
        # ElasticTransform(p=0.1),
        ToGray(p=0.05),
    ],
        bbox_params={'format': 'pascal_voc',
                        'min_area': 1,
                        'min_visibility': 0.1,
                        'label_fields': ['labels']},
    p=p)


def preprocess_image(img):
    from keras.applications.densenet import preprocess_input
    return preprocess_input(img)


def read_cropped_image(p, x0, y0, x1, y1, augment, img_shape=(224, 224, 3)):
    anisotropy = 2.15  # The horizontal compression ratio
    if augment:
        crop_margin = random.uniform(0.01, 0.09)
    else:
        crop_margin = 0.05

    # Read the image
    img = read_single_image(p)
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
        try:
            augm = strong_aug(p=1.0)(image=img, bboxes=bb, labels=['labels'])
        except:
            print('Error albumentations: {}'.format(os.path.basename(p)))
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
    try:
        img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    except:
        print('Resize error! Image shape: {}'.format(img.shape), p, x0, y0, x1, y1)
        img = read_single_image(p)
        img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    if len(img.shape) == 2:
        img = np.concatenate((img, img, img), axis=2)

    return img


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


if __name__ == '__main__':
    start_time = time.time()

    if 0:
        model, branch_model, head_model = build_model(0.001, 0.0, img_shape=(224, 224, 3))
        print(model.summary())
        print(branch_model.summary())
        print(head_model.summary())

    bboxes = get_boxes()
    img_id = '00fee3975.jpg'
    bb = bboxes[img_id]
    for i in range(10):
        img = read_cropped_image(INPUT_PATH + 'train/' + img_id, bb[0], bb[1], bb[2], bb[3], False)
        show_image(img, type='rgb')

    print('Time: {:.0f} sec'.format(time.time() - start_time))

