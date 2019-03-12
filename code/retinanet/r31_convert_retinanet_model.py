#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

if __name__ == '__main__':
    import os
    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import argparse
import sys
from a00_common_functions import *


# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from keras_retinanet import models


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

    parser.add_argument('model_in', help='The model to convert.')
    parser.add_argument('model_out', help='Path to save the converted model to.')
    parser.add_argument('--backbone', help='The backbone of the model to convert.', default='resnet50')
    parser.add_argument('--no-nms', help='Disables non maximum suppression.', dest='nms', action='store_false')
    parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.', dest='class_specific_filter', action='store_false')

    return parser.parse_args(args)


def main(args=None):
    from keras_retinanet.utils.config import parse_anchor_parameters

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = dict()
    config['anchor_parameters'] = dict()
    config['anchor_parameters']['sizes'] = '16 32 64 128 256 512'
    config['anchor_parameters']['strides'] = '8 16 32 64 128'
    config['anchor_parameters']['ratios'] = '0.1 0.5 1 2 4 8'
    config['anchor_parameters']['scales'] = '1 1.25 1.5 1.75'
    anchor_params = None
    anchor_params = parse_anchor_parameters(config)

    # load and convert model
    model = models.load_model(args.model_in,  backbone_name=args.backbone)
    model = models.convert_model(model, nms=args.nms, class_specific_filter=args.class_specific_filter, anchor_params=anchor_params)

    # save model
    model.save(args.model_out)


if __name__ == '__main__':
    params = [
        MODELS_PATH + 'retinanet2/resnet152_fold_1_last.h5',
        MODELS_PATH + 'retinanet2/resnet152_fold_1_last_converted.h5',
        '--backbone', 'resnet152'
    ]
    main(params)
