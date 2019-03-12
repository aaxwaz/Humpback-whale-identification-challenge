# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *


def merge_boxes_v1():
    box1 = load_from_file_fast(OUTPUT_PATH + 'p2bb_v5.pkl')
    box2 = load_from_file_fast(OUTPUT_PATH + 'p2bb_retinanet_resnet152_cropping.pkl')
    print(len(box1))
    print(len(box2))

    # Only use IDs from first boxes
    for el in box1:
        if el in box2:
            b1 = list(box1[el])
            b2 = list(box2[el])
            new_box0 = int(round((b1[0] + b2[0]) / 2))
            new_box1 = int(round((b1[1] + b2[1]) / 2))
            new_box2 = int(round((b1[2] + b2[2]) / 2))
            new_box3 = int(round((b1[3] + b2[3]) / 2))
            if 0:
                path = INPUT_PATH + 'test/' + el + '.jpg'
                try:
                    img = read_image_bgr_fast(path)
                except:
                    path = INPUT_PATH + 'train/' + el + '.jpg'
                    img = read_image_bgr_fast(path)
                img = cv2.rectangle(img.copy(), (b1[0], b1[1]), (b1[2], b1[3]), (0, 0, 255), 2)
                img = cv2.rectangle(img.copy(), (b2[0], b2[1]), (b2[2], b2[3]), (0, 255, 0), 2)
                img = cv2.rectangle(img.copy(), (new_box0, new_box1), (new_box2, new_box3), (255, 0, 0), 2)
                show_image(img, type='bgr')

            b1[0] = new_box0
            b1[1] = new_box1
            b1[2] = new_box2
            b1[3] = new_box3
            box1[el] = tuple(b1)

    save_in_file_fast(box1, OUTPUT_PATH + 'p2bb_averaged_v1.pkl')


def merge_boxes_v2():
    box1 = load_from_file_fast(OUTPUT_PATH + 'p2bb_playground_v2.pkl')
    box2 = load_from_file_fast(OUTPUT_PATH + 'p2bb_retinanet_resnet152_cropping_playground.pkl')
    print(len(box1))
    print(len(box2))

    # Only use IDs from first boxes
    for el in box1:
        if el in box2:
            b1 = list(box1[el])
            b2 = list(box2[el])
            new_box0 = int(round((b1[0] + b2[0]) / 2))
            new_box1 = int(round((b1[1] + b2[1]) / 2))
            new_box2 = int(round((b1[2] + b2[2]) / 2))
            new_box3 = int(round((b1[3] + b2[3]) / 2))
            if 0:
                path = INPUT_PATH + 'playground/test/' + el + '.jpg'
                try:
                    img = read_image_bgr_fast(path)
                except:
                    path = INPUT_PATH + 'playground/train/' + el + '.jpg'
                    img = read_image_bgr_fast(path)
                img = cv2.rectangle(img.copy(), (b1[0], b1[1]), (b1[2], b1[3]), (0, 0, 255), 2)
                img = cv2.rectangle(img.copy(), (b2[0], b2[1]), (b2[2], b2[3]), (0, 255, 0), 2)
                img = cv2.rectangle(img.copy(), (new_box0, new_box1), (new_box2, new_box3), (255, 0, 0), 2)
                show_image(img, type='bgr')

            b1[0] = new_box0
            b1[1] = new_box1
            b1[2] = new_box2
            b1[3] = new_box3
            box1[el] = tuple(b1)

    # Use original boxes
    in1 = open(OUTPUT_PATH + 'retinanet/cropping_train_v2.csv')
    while 1:
        line = in1.readline().strip()
        if line == '':
            break
        arr = line.split(',')
        name = os.path.basename(arr[0])
        x1, y1, x2, y2 = arr[1:5]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        if 0:
            img = read_single_image(INPUT_PATH + 'playground/train/' + name)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            show_image(img)
        box1[name[:-4]] = (x1, y1, x2, y2)
    in1.close()

    save_in_file_fast(box1, OUTPUT_PATH + 'p2bb_averaged_playground_v1.pkl')


if __name__ == '__main__':
    merge_boxes_v1()
    merge_boxes_v2()
