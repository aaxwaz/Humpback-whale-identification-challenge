# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *


RETINANET_PATH = OUTPUT_PATH + 'retinanet/'
if not os.path.isdir(RETINANET_PATH):
    os.mkdir(RETINANET_PATH)


def create_single_csv(out_path, train_files, bboxes):
    out = open(out_path, 'w')
    # out.write('img_path,x1,y1,x2,y2,class_name\n')
    for id in train_files:
        name = id[:-4]
        if name not in bboxes:
            continue
        full_path = INPUT_PATH + 'train/' + id
        x1, y1, x2, y2 = bboxes[name]
        if x2 < x1 or y2 < y1:
            print('Strange: {} {}'.format(id, bboxes[name]))
            img = read_single_image(full_path)
            show_image(img)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            show_image(img)
        out.write("{},{},{},{},{},whale\n".format(full_path, x1, y1, x2, y2))

    out.close()


def create_csv_retinanet(nfolds):
    bboxes = load_from_file_fast(OUTPUT_PATH + 'p2bb_v5.pkl')
    kfold_split = get_kfold_split_retinanet(nfolds)

    fold_num = 0
    for train_files, valid_files in kfold_split:
        fold_num += 1
        if 1:
            out_path = RETINANET_PATH + 'fold_{}_train.csv'.format(fold_num)
            create_single_csv(out_path, train_files, bboxes)
        if 1:
            out_path = RETINANET_PATH + 'fold_{}_valid.csv'.format(fold_num)
            create_single_csv(out_path, valid_files, bboxes)


def create_classes_file():
    f = open(OUTPUT_PATH + 'retinanet/classes.txt', 'w')
    f.write('whale,0\n')
    f.close()


if __name__ == '__main__':
    nfolds = 5
    create_csv_retinanet(nfolds)
    create_classes_file()
