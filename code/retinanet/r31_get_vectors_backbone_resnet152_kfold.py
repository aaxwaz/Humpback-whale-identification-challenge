# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os
    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
from retinanet.a01_ensemble_boxes_functions import *
import cv2
import matplotlib.pyplot as plt
import pydicom
from keras_retinanet import models
from shutil import copy2


RETINA_PREDICTIONS_TRAIN = OUTPUT_PATH + 'cache_retinanet_resnet152_kfold_train/'
if not os.path.isdir(RETINA_PREDICTIONS_TRAIN):
    os.mkdir(RETINA_PREDICTIONS_TRAIN)

RETINA_PREDICTIONS_TEST = OUTPUT_PATH + 'cache_retinanet_resnet152_kfold_test/'
if not os.path.isdir(RETINA_PREDICTIONS_TEST):
    os.mkdir(RETINA_PREDICTIONS_TEST)

RETINA_PREDICTIONS_PL_TRAIN = OUTPUT_PATH + 'cache_retinanet_resnet152_kfold_playground_train/'
if not os.path.isdir(RETINA_PREDICTIONS_PL_TRAIN):
    os.mkdir(RETINA_PREDICTIONS_PL_TRAIN)

RETINA_PREDICTIONS_PL_TEST = OUTPUT_PATH + 'cache_retinanet_resnet152_kfold_playground_test/'
if not os.path.isdir(RETINA_PREDICTIONS_PL_TEST):
    os.mkdir(RETINA_PREDICTIONS_PL_TEST)


def show_image_debug(draw, boxes, scores, labels):
    from keras_retinanet.utils.visualization import draw_box, draw_caption
    from keras_retinanet.utils.colors import label_color

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.4:
            break

        color = (0, 255, 0)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format('target', score)
        draw_caption(draw, b, caption)
    draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
    show_image(draw)


def get_retinanet_predictions(model, image):
    from keras_retinanet.utils.image import preprocess_image, resize_image

    show_debug_images = False
    show_mirror_predictions = False

    if show_debug_images:
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side=600, max_side=800)

    # Add mirror
    if 1:
        image = np.stack((image, image[:, ::-1, :]), axis=0)
    else:
        image = np.array([image])

    # process image
    start = time.time()
    print('Image shape: {} Scale: {}'.format(image.shape, scale))
    boxes, scores, labels = model.predict_on_batch(image)
    print('Detections shape: {} {} {}'.format(boxes.shape, scores.shape, labels.shape))
    print("Processing time: {:.2f} sec".format(time.time() - start))

    if show_debug_images:
        if show_mirror_predictions:
            draw = draw[:, ::-1, :]
        boxes_init = boxes.copy()
        boxes_init /= scale

    boxes[:, :, 0] /= image.shape[2]
    boxes[:, :, 2] /= image.shape[2]
    boxes[:, :, 1] /= image.shape[1]
    boxes[:, :, 3] /= image.shape[1]

    if show_debug_images:
        if show_mirror_predictions:
            show_image_debug(draw.astype(np.uint8), boxes_init[1:], scores[1:], labels[1:])
        else:
            show_image_debug(draw.astype(np.uint8), boxes_init[:1], scores[:1], labels[:1])

    return boxes, scores, labels


def get_retinanet_predictions_for_train(model, fold):
    restore_from_cache = 1
    thr = 0.5
    dice_avg = 0
    count_avg = 0
    kfold_split = get_kfold_split_retinanet(5)

    res = dict()
    out_pkl = OUTPUT_PATH + "boxes/retinanet_resnet152_boxes_{}_fold_{}.pkl".format('train', fold)

    train_files, valid_files = kfold_split[fold-1]
    train_df = pd.read_csv(INPUT_PATH + 'train.csv')
    train_df = train_df[train_df['Image'].isin(valid_files)]
    full_train_ids = train_df['Image'].values
    bboxes = load_from_file_fast(OUTPUT_PATH + 'p2bb_v5.pkl')

    bad_boxes = 0
    no_bbox_images = 0
    fold_independent_dice = 0
    fold_independent_dice_count = 0

    for i in range(len(full_train_ids)):
        name = full_train_ids[i][:-4]
        print('Go for {}'.format(name))
        cache_path = RETINA_PREDICTIONS_TRAIN + name + '.pklz'
        path = INPUT_PATH + 'train/' + name + '.jpg'
        img = read_image_bgr_fast(path)
        if not os.path.isfile(cache_path) or restore_from_cache == 0:
            boxes, scores, labels = get_retinanet_predictions(model, img)
            save_in_file((boxes, scores, labels), cache_path)
        else:
            boxes, scores, labels = load_from_file(cache_path)

        # print(boxes, scores, labels)

        filtered_boxes = filter_boxes(boxes, scores, labels, 0.01)
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, 0.55, 'avg')
        if len(merged_boxes) > 4:
            # sort by score
            merged_boxes = np.array(merged_boxes)
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:4]

        print(img.shape)
        if name in bboxes:
            real_box = bboxes[name]
        else:
            real_box = (0, 0, 1, 1)
        print((real_box[0], real_box[1]), (real_box[2], real_box[3]))

        if len(merged_boxes) > 0:
            box = merged_boxes[0][2:]
            box[0] *= img.shape[1]
            box[1] *= img.shape[0]
            box[2] *= img.shape[1]
            box[3] *= img.shape[0]
            box = box.astype(np.int32)
            res[name] = list(box)

            print((box[0], box[1]), (box[2], box[3]))
            img = cv2.rectangle(img.copy(), (real_box[0], real_box[1]), (real_box[2], real_box[3]), (0, 0, 255), 2)
            img = cv2.rectangle(img.copy(), (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # show_image(img, type='bgr')

            d = bb_intersection_over_union(box, real_box)
            print('IOU for image {}: {:.6f}'.format(name, d))
            if d < 0.8:
                show_image(img, type='bgr')
                bad_boxes += 1
                # copy2(path, OUTPUT_PATH + 'for_annotation4/')
                # output_single_xml_label(OUTPUT_PATH + 'for_annotation4/', name, box[0], box[1], box[2], box[3])
                # cv2.imwrite(OUTPUT_PATH + 'debug/' + name + '.jpg', img)
        else:
            img = cv2.rectangle(img.copy(), (real_box[0], real_box[1]), (real_box[2], real_box[3]), (0, 0, 255), 2)
            # show_image(img, type='bgr')
            d = 0
            no_bbox_images += 1

        dice_avg += d
        count_avg += 1
        fold_independent_dice += d
        fold_independent_dice_count += 1

    print('BBox bad: {}'.format(bad_boxes))
    print('BBox not found: {}'.format(no_bbox_images))
    print('Avg IOU: {:.6f}'.format(fold_independent_dice / fold_independent_dice_count))

    save_in_file_fast(res, out_pkl)
    return dice_avg / count_avg, thr


def get_retinanet_predictions_for_others(model, fold, type='test'):
    restore_from_cache = 1

    res = dict()
    out_pkl = OUTPUT_PATH + "boxes/retinanet_resnet152_boxes_{}_fold_{}.pkl".format(type, fold)
    if type == 'test':
        print('Test')
        files = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
        ids = files['Image'].values
        out_dir = RETINA_PREDICTIONS_TEST
        in_path = INPUT_PATH + 'test/'
    elif type == 'playground_train':
        print('Playground train')
        files = pd.read_csv(INPUT_PATH + 'playground/train.csv')
        ids = files['Image'].values
        out_dir = RETINA_PREDICTIONS_PL_TRAIN
        in_path = INPUT_PATH + 'playground/train/'
    elif type == 'playground_test':
        print('Playground test')
        files = pd.read_csv(INPUT_PATH + 'playground/sample_submission.csv')
        ids = files['Image'].values[::-1]
        out_dir = RETINA_PREDICTIONS_PL_TEST
        in_path = INPUT_PATH + 'playground/test/'

    no_bbox_images = 0

    for i in range(len(ids)):
        name = ids[i][:-4]
        print('Go for {}'.format(name))
        cache_path = out_dir + name + '_fold_{}.pklz'.format(fold)
        path = in_path + name + '.jpg'
        img = read_image_bgr_fast(path)
        if not os.path.isfile(cache_path) or restore_from_cache == 0:
            boxes, scores, labels = get_retinanet_predictions(model, img)
            save_in_file((boxes, scores, labels), cache_path)
        else:
            boxes, scores, labels = load_from_file(cache_path)

        # print(boxes, scores, labels)

        filtered_boxes = filter_boxes(boxes, scores, labels, 0.01)
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, 0.55, 'avg')
        if len(merged_boxes) > 4:
            # sort by score
            merged_boxes = np.array(merged_boxes)
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:4]

        print(img.shape)
        if len(merged_boxes) > 0:
            box = merged_boxes[0][2:]
            box[0] *= img.shape[1]
            box[1] *= img.shape[0]
            box[2] *= img.shape[1]
            box[3] *= img.shape[0]
            box = box.astype(np.int32)
            res[name] = list(box)
            # img = cv2.rectangle(img.copy(), (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # show_image(img, type='bgr')
        else:
            # copy2(path, OUTPUT_PATH + '/for_annotation/')
            no_bbox_images += 1

    print('BBox not found: {}'.format(no_bbox_images))
    save_in_file_fast(res, out_pkl)


def create_bboxes_files():
    bboxes = dict()

    files = glob.glob(RETINA_PREDICTIONS_TRAIN + '*.pklz')
    for f in files:
        name = os.path.basename(f)[:-5]
        boxes, scores, labels = load_from_file(f)

        filtered_boxes = filter_boxes(boxes, scores, labels, 0.01)
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, 0.55, 'avg')
        if len(merged_boxes) > 4:
            # sort by score
            merged_boxes = np.array(merged_boxes)
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:4]

        if len(merged_boxes) > 0:
            img_path = INPUT_PATH + 'train/' + name + '.jpg'
            img = pyvips.Image.new_from_file(img_path, access='sequential')
            w, h = img.width, img.height
            box = merged_boxes[0][2:]
            box[0] *= w
            box[1] *= h
            box[2] *= w
            box[3] *= h
            box = box.astype(np.int32)
            bboxes[name] = tuple(box)
            print((box[0], box[1]), (box[2], box[3]))

    files = glob.glob(RETINA_PREDICTIONS_TEST + '*.pklz')
    for f in files:
        name = os.path.basename(f)[:-5]
        boxes, scores, labels = load_from_file(f)

        filtered_boxes = filter_boxes(boxes, scores, labels, 0.01)
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, 0.55, 'avg')
        if len(merged_boxes) > 4:
            # sort by score
            merged_boxes = np.array(merged_boxes)
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:4]

        if len(merged_boxes) > 0:
            img_path = INPUT_PATH + 'test/' + name + '.jpg'
            img = pyvips.Image.new_from_file(img_path, access='sequential')
            w, h = img.width, img.height
            box = merged_boxes[0][2:]
            box[0] *= w
            box[1] *= h
            box[2] *= w
            box[3] *= h
            box = box.astype(np.int32)
            bboxes[name] = tuple(box)
            print((box[0], box[1]), (box[2], box[3]))

    save_in_file_fast(bboxes, OUTPUT_PATH + 'p2bb_retinanet_resnet152_cropping.pkl')
    print('BBoxes: {}'.format(len(bboxes)))

    bboxes = dict()
    files = glob.glob(RETINA_PREDICTIONS_PL_TRAIN + '*.pklz')
    for f in files:
        name = os.path.basename(f)[:-5]
        boxes, scores, labels = load_from_file(f)

        filtered_boxes = filter_boxes(boxes, scores, labels, 0.01)
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, 0.55, 'avg')
        if len(merged_boxes) > 4:
            # sort by score
            merged_boxes = np.array(merged_boxes)
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:4]

        if len(merged_boxes) > 0:
            img_path = INPUT_PATH + 'playground/train/' + name + '.jpg'
            img = pyvips.Image.new_from_file(img_path, access='sequential')
            w, h = img.width, img.height
            box = merged_boxes[0][2:]
            box[0] *= w
            box[1] *= h
            box[2] *= w
            box[3] *= h
            box = box.astype(np.int32)
            bboxes[name] = tuple(box)
            print((box[0], box[1]), (box[2], box[3]))

    files = glob.glob(RETINA_PREDICTIONS_PL_TEST + '*.pklz')
    for f in files:
        name = os.path.basename(f)[:-5]
        boxes, scores, labels = load_from_file(f)

        filtered_boxes = filter_boxes(boxes, scores, labels, 0.01)
        merged_boxes = merge_all_boxes_for_image(filtered_boxes, 0.55, 'avg')
        if len(merged_boxes) > 4:
            # sort by score
            merged_boxes = np.array(merged_boxes)
            merged_boxes = merged_boxes[merged_boxes[:, 1].argsort()[::-1]][:4]

        if len(merged_boxes) > 0:
            img_path = INPUT_PATH + 'playground/test/' + name + '.jpg'
            img = pyvips.Image.new_from_file(img_path, access='sequential')
            w, h = img.width, img.height
            box = merged_boxes[0][2:]
            box[0] *= w
            box[1] *= h
            box[2] *= w
            box[3] *= h
            box = box.astype(np.int32)
            bboxes[name] = tuple(box)
            print((box[0], box[1]), (box[2], box[3]))

    save_in_file_fast(bboxes, OUTPUT_PATH + 'p2bb_retinanet_resnet152_cropping_playground.pkl')
    print('BBoxes: {}'.format(len(bboxes)))


if __name__ == '__main__':
    fold = 1
    model = models.load_model(MODELS_PATH + 'retinanet2/resnet152_fold_{}_last_converted.h5'.format(fold), backbone_name='resnet152')
    get_retinanet_predictions_for_train(model, fold)
    get_retinanet_predictions_for_others(model, fold, 'test')
    get_retinanet_predictions_for_others(model, fold, 'playground_train')
    get_retinanet_predictions_for_others(model, fold, 'playground_test')
    create_bboxes_files()

'''
Fold 1: Avg IOU: 0.927350 BBox not found: 2
'''