from keras_retinanet.preprocessing.generator import Generator
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.image import (
    adjust_transform_for_image,
    apply_transform,
)
from keras_retinanet.utils.transform import transform_aabb
from keras_retinanet.utils.anchors import (
    anchors_for_shape,
)
from keras_retinanet.utils.config import parse_anchor_parameters

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
import random
import cv2


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """ Parse the classes file given by csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def fix_wide_images_v1(image, bboxes):
    limit_scale = 1.0
    scale_ratio = image.shape[1] / image.shape[0]
    if scale_ratio > limit_scale:
        # random_scale = random.uniform(limit_scale, scale_ratio)
        random_scale = scale_ratio
        # print('Random scale: {}'.format(random_scale))
        image = cv2.resize(image, (int(image.shape[1] / random_scale), image.shape[0]), interpolation=cv2.INTER_CUBIC)
        for i in range(len(bboxes)):
            bboxes[i][0] = np.round(bboxes[i][0] / random_scale)
            bboxes[i][2] = np.round(bboxes[i][2] / random_scale)

    return image, bboxes


def fix_wide_images(image, bboxes):
    scale_ratio_x = image.shape[1] / 600
    scale_ratio_y = image.shape[0] / 600
    image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_CUBIC)
    for i in range(len(bboxes)):
        bboxes[i][0] = np.round(bboxes[i][0] / scale_ratio_x)
        bboxes[i][1] = np.round(bboxes[i][1] / scale_ratio_y)
        bboxes[i][2] = np.round(bboxes[i][2] / scale_ratio_x)
        bboxes[i][3] = np.round(bboxes[i][3] / scale_ratio_y)

    return image, bboxes


class CSVGeneratorWhales(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        csv_data_file,
        csv_class_file,
        base_dir=None,
        **kwargs
    ):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []
        self.image_data  = {}
        self.base_dir    = base_dir

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)

        # parse the provided class file
        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                self.image_data = _read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.image_names = list(self.image_data.keys())

        super(CSVGeneratorWhales, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        path        = self.image_names[image_index]
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        for idx, annot in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                float(annot['x1']),
                float(annot['y1']),
                float(annot['x2']),
                float(annot['y2']),
            ]]))

        return annotations

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets


    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)


    def preprocess_group_entry(self, image, annotations):
        """ Preprocess image and its annotations.
        """

        # Special whales case for very wide images try to scale it!
        # image, annotations['bboxes'] = fix_wide_images(image, annotations['bboxes'])

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # preprocess the image
        image = self.preprocess_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        if 0:
            from a00_common_functions import show_image
            print(image.shape, annotations)
            show_image(image)

        return image, annotations


    def preprocess_group(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group


    def random_transform_group_entry_v1(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations


    def random_transform_group_entry(self, image, annotations, transform=None):
        """ Randomly transforms image and annotation.
        """
        debug = False
        # randomly transform both image and annotations
        if debug:
            import cv2
            from a00_common_functions import show_image
            image1 = image.copy()
            print(image1.min(), image1.max())
            print(annotations)
            for b in annotations['bboxes']:
                b = b.astype(np.int32)
                image1 = cv2.rectangle(image1, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            show_image(image1)

        if self.transform_generator:
            ann = dict()
            ann['image'] = image.copy()
            ann['labels'] = annotations['labels'].copy()
            ann['bboxes'] = annotations['bboxes'].copy()
            augm = self.transform_generator(**ann)
            image = augm['image']
            annotations['bboxes'] = np.array(augm['bboxes'])

        if debug:
            image1 = image.copy()
            print(annotations)
            for b in annotations['bboxes']:
                b = b.astype(np.int32)
                image1 = cv2.rectangle(image1, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            show_image(image1)

        return image, annotations

    def generate_anchors(self, image_shape):
        anchor_params = None
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = parse_anchor_parameters(self.config)
        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        anchors   = self.generate_anchors(max_shape)

        batches = self.compute_anchor_targets(
            anchors,
            image_group,
            annotations_group,
            self.num_classes()
        )

        return list(batches)