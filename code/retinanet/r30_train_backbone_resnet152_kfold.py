import argparse
import os
import sys
import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import layers  # noqa: F401
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.config import parse_anchor_parameters

from a00_common_functions import *
from retinanet.a01_whales_generator import CSVGeneratorWhales
from retinanet.a02_eval import Evaluate_IOU
from albumentations import *


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.
    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.
    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.
    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        evaluation = Evaluate_IOU(validation_generator, tensorboard=tensorboard_callback)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)


    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_fold_{fold}_last.h5'.format(backbone=args.backbone, fold=args.fold)
            ),
            verbose=1,
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_fold_{fold}_{{epoch:02d}}_{{IOU:.4f}}.h5'.format(backbone=args.backbone, fold=args.fold)
            ),
            verbose=1,
            save_best_only=True,
            monitor="IOU",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'loss',
        factor   = 0.9,
        patience = 5,
        verbose  = 1,
        mode     = 'auto',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
    ))

    return callbacks


def create_generators(args, preprocess_image):
    global config

    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'preprocess_image' : preprocess_image,
    }

    if 0:
        # create random transform generator for augmenting training data
        transform_generator = random_transform_generator(
            min_rotation=-0.2,
            max_rotation=0.2,
            min_translation=(-0.2, -0.2),
            max_translation=(0.2, 0.2),
            min_shear=-0.2,
            max_shear=0.2,
            min_scaling=(0.8, 0.8),
            max_scaling=(1.2, 1.2),
            flip_x_chance=0.5,
            flip_y_chance=0.0,
        )
    else:
        if 1:
            transform_generator = Compose([
                HorizontalFlip(p=0.5),
                # VerticalFlip(p=0.5),
                # RandomRotate90(p=0.5),
                OneOf([
                    IAAAdditiveGaussianNoise(),
                    GaussNoise(),
                ], p=0.2),
                OneOf([
                    MotionBlur(p=.1),
                    MedianBlur(blur_limit=3, p=.1),
                    Blur(blur_limit=3, p=.1),
                ], p=0.2),
                # ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=10, p=0.1),
                # IAAPiecewiseAffine(p=0.05),
                OneOf([
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(),
                ], p=0.2),
                OneOf([
                    RGBShift(p=1.0, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
                    HueSaturationValue(p=1.0),
                ], p=0.5),
                ToGray(p=0.3),
                JpegCompression(p=0.3, quality_lower=25, quality_upper=99),
            ], bbox_params={'format': 'pascal_voc',
                            'min_area': 1,
                            'min_visibility': 0.1,
                            'label_fields': ['labels']}, p=1.0)
        else:
            transform_generator = Compose([
                HorizontalFlip(p=0.5),
            ], bbox_params={'format': 'pascal_voc',
                            'min_area': 1,
                            'min_visibility': 0.1,
                            'label_fields': ['labels']}, p=1.0)

    train_generator = CSVGeneratorWhales(
        args.annotations,
        args.classes,
        transform_generator=transform_generator,
        config=config,
        **common_args
    )

    if args.val_annotations:
        validation_generator = CSVGeneratorWhales(
            args.val_annotations,
            args.classes,
            config=config,
            **common_args
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.
    Args
        parsed_args: parser.parse_args()
    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--fixed-labels', help='Use the exact specified labels.', default=False)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',       help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--fold',            help='Fold number.', type=int, default=1)
    parser.add_argument('--steps',           help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',   help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config', help='Path to a configuration parameters .ini file.')

    return check_args(parser.parse_args(args))


def main(args=None):
    global config
    from keras import backend as K

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    print('Arguments: {}'.format(args))

    # create object that stores backbone information
    backbone = models.backbone(args.backbone)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model = model
        anchor_params = None
        if 'anchor_parameters' in config:
            anchor_params = parse_anchor_parameters(config)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            config=config
        )

    # print model summary
    print(model.summary())

    print('Learning rate: {}'.format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print('Learning rate: {}'.format(K.get_value(model.optimizer.lr)))

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    init_epoch = 0
    if args.snapshot:
        init_epoch = int(args.snapshot.split("_")[-2])
    print('Init epoch: {}'.format(init_epoch))

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=2,
        callbacks=callbacks,
        initial_epoch=init_epoch,
    )


if __name__ == '__main__':
    config = dict()
    config['anchor_parameters'] = dict()
    config['anchor_parameters']['sizes'] = '16 32 64 128 256 512'
    config['anchor_parameters']['strides'] = '8 16 32 64 128'
    config['anchor_parameters']['ratios'] = '0.1 0.5 1 2 4 8'
    config['anchor_parameters']['scales'] = '1 1.25 1.5 1.75'

    fold = 1
    root_path = ROOT_PATH
    params = [
        # '--snapshot', root_path + 'models/retinanet1/resnet152_fold_0_226_0.8790.h5',
        # '--imagenet-weights',
        # '--weights', '../../weights/retinanet_resnet152_level_1_v1.2.h5',
        '--gpu', '0',
        '--steps', '10000',
        '--snapshot-path', root_path + 'models/retinanet2/',
        # '--multi-gpu', '2',
        # '--multi-gpu-force',
        # '--backbone', 'mobilenet224_1.0',
        '--backbone', 'resnet152',
        '--batch-size', '1',
        '--image-min-side', '600',
        '--image-max-side', '800',
        '--fold', '{}'.format(fold),
        'csv',
        root_path + 'modified_data/retinanet/fold_{}_train.csv'.format(fold),
        root_path + 'modified_data/retinanet/classes.txt',
        '--val-annotations', root_path + 'modified_data/retinanet/fold_{}_valid.csv'.format(fold),
    ]
    main(params)

'''
Fold 1:
Ep 12: 2556s - loss: 0.3100 - regression_loss: 0.2739 - classification_loss: 0.0360 IOU: 0.8636
Ep 13: 2550s - loss: 0.2982 - regression_loss: 0.2640 - classification_loss: 0.0342 IOU: 0.8567
Ep 14: 2558s - loss: 0.2975 - regression_loss: 0.2636 - classification_loss: 0.0339 IOU: 0.8614
Ep 15: 2559s - loss: 0.2912 - regression_loss: 0.2591 - classification_loss: 0.0321 IOU: 0.8711 Empty detections: 10 [0.19%]
Ep 16: 2550s - loss: 0.2817 - regression_loss: 0.2520 - classification_loss: 0.0297 IOU: 0.8558
Ep 17: 2558s - loss: 0.2719 - regression_loss: 0.2428 - classification_loss: 0.0292 IOU: 0.8604 Empty detections: 6 [0.12%]
Ep 18: 2550s - loss: 0.2692 - regression_loss: 0.2413 - classification_loss: 0.0279 IOU: 0.8609
Ep 19: 2585s - loss: 0.2497 - regression_loss: 0.2265 - classification_loss: 0.0232 IOU: 0.8730
Ep 23: 2556s - loss: 0.2425 - regression_loss: 0.2199 - classification_loss: 0.0226 IOU: 0.8809
Ep 25: 2562s - loss: 0.2379 - regression_loss: 0.2171 - classification_loss: 0.0207 IOU: 0.8746 Empty detections: 7 [0.13%]
Ep 32: 2567s - loss: 0.2102 - regression_loss: 0.1937 - classification_loss: 0.0165 IOU: 0.8886 Empty detections: 3 [0.06%]
'''