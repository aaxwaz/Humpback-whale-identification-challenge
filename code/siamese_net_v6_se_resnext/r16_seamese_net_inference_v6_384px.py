# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

if __name__ == '__main__':
    import os

    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_common_functions import *
from siamese_net_v6_se_resnext.a01_seamese_net_models_v6 import read_cropped_image, build_model, preprocess_image
from siamese_net_v6_se_resnext.r15_seamese_net_train_v6_finetune_384px import get_tagged_data, get_boxes, expand_path, get_kfold_split_weimin


BOX_SIZE = 384
CACHE_PATH = OUTPUT_PATH + 'seamese_net_v6_rgb_seresnext50_384px/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)


def process_train_images_with_branch_model(branch_model, fold):
    cache_path = CACHE_PATH + 'branch_res_train_fold_{}.pklz'.format(fold)
    if not os.path.isfile(cache_path):
        print('Go branch model train...')
        start_time = time.time()
        bboxes = get_boxes()
        CLASSES = get_classes_array()
        tagged = get_tagged_data()
        train_ids = []
        labels = dict()
        train_size = len(tagged)
        missed = 0
        # train_size = 10
        train_res = []
        i = 0
        for image_id in sorted(list(tagged.keys())):
            clss = tagged[image_id]
            # print('Go for: {}'.format(image_id))
            f = expand_path(image_id)
            lbl = CLASSES.index(clss)
            if image_id not in bboxes:
                print('Missed bbox for {}!'.format(image_id))
                missed += 1
                continue
            bb = bboxes[image_id]
            labels[image_id] = lbl
            train_ids.append(image_id)
            img = read_cropped_image(f, bb[0], bb[1], bb[2], bb[3], False, img_shape=(BOX_SIZE, BOX_SIZE, 3))
            img = np.expand_dims(img, axis=0)
            img = preprocess_image(img.astype(np.float32))
            preds = branch_model.predict(img)
            train_res.append(preds[0])
            i += 1

        train_res = np.array(train_res)
        train_ids = np.array(train_ids)

        print('Train preds: {}'.format(train_res.shape))
        print('Missed bboxes: {}'.format(missed))
        print('Read train time: {:.0f} sec'.format(time.time() - start_time))
        save_in_file((train_res, train_ids, labels), cache_path)
    else:
        print('Restore train from cache: {}'.format(cache_path))
        train_res, train_ids, labels = load_from_file(cache_path)

    return train_res, train_ids, labels


def process_tst_images_with_branch_model(branch_model, fold):
    cache_path = CACHE_PATH + 'branch_res_test_fold_{}.pklz'.format(fold)
    if not os.path.isfile(cache_path):
        print('Go branch model test...')
        start_time = time.time()
        bboxes = get_boxes()
        missed = 0
        test_df = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
        test_ids = []
        test_size = len(test_df)
        # test_size = 10
        test_res = []
        i = 0
        for index, row in test_df.iterrows():
            # print('Go for: {}'.format(row['Image']))
            f = INPUT_PATH + 'test/' + row['Image']
            image_id = row['Image']
            if image_id in bboxes:
                bb = bboxes[image_id]
            else:
                print('Missed bbox for {}!'.format(image_id))
                missed += 1
                img = read_single_image(f)
                bb = (0, 0, img.shape[1], img.shape[0])
            img = read_cropped_image(f, bb[0], bb[1], bb[2], bb[3], False, img_shape=(BOX_SIZE, BOX_SIZE, 3))
            img = np.expand_dims(img, axis=0)
            img = preprocess_image(img.astype(np.float32))
            preds = branch_model.predict(img)
            test_res.append(preds[0])
            test_ids.append(row['Image'])
            i += 1
        test_res = np.array(test_res)
        test_ids = np.array(test_ids)

        print('Test preds: {}'.format(test_res.shape))
        print('Missed bboxes: {}'.format(missed))
        print('Read test time: {:.0f} sec'.format(time.time() - start_time))
        save_in_file((test_res, test_ids), cache_path)
    else:
        print('Restore test from cache: {}'.format(cache_path))
        test_res, test_ids = load_from_file(cache_path)

    return test_res, test_ids


def check_model(model, branch_model, head_model, img1, img2):
    arr = [img1, img2]
    preds1 = model.predict(arr)
    print(preds1.shape)
    preds2 = branch_model.predict(img1)
    preds3 = branch_model.predict(img2)
    preds4 = head_model.predict([preds2, preds3])
    print(preds1, preds4)


def get_single_score_max(preds, real_answ, compare_ids, real_labels, CLASSES, thr):
    print_num = 5
    verbose = False

    # print(preds.shape)
    compare_ids = np.array(compare_ids)
    real_labels = np.array(real_labels)
    sorted_args = np.argsort(preds[:, 0])[::-1]

    probs = preds[sorted_args][:, 0]
    ids = compare_ids[sorted_args]
    train_labels = real_labels[sorted_args]

    # Remove same id
    cond = np.where(~np.isin(ids, [id]))
    probs = probs[cond]
    ids = ids[cond]
    train_labels = train_labels[cond]

    if verbose:
        print(probs[:print_num])
        print(ids[:print_num])
        print(train_labels[:print_num])
        train_names = []
        for t in train_labels[:10]:
            train_names.append(CLASSES[t])
        print(train_names)

    answ = []
    cond = probs > thr
    prob1 = probs[cond]
    train_labels1 = train_labels[cond]

    for j in range(min(len(prob1), 5)):
        lbl = train_labels1[j]
        if CLASSES[lbl] not in answ:
            answ.append(CLASSES[lbl])

    if len(answ) < 5:
        answ.append(CLASSES[0])

    if len(answ) < 5:
        for j, p in enumerate(probs):
            lbl = train_labels[j]
            if CLASSES[lbl] not in answ:
                answ.append(CLASSES[lbl])
            if len(answ) >= 5:
                break

    if verbose:
        print(str(answ))

    score = apk([real_answ], answ, k=5)
    return score, answ


def get_single_score(preds, real_answ, compare_ids, real_labels, CLASSES, thr):
    print_num = 5
    verbose = False

    # print(preds.shape)
    compare_ids = np.array(compare_ids)
    real_labels = np.array(real_labels)
    sorted_args = np.argsort(preds[:, 0])[::-1]

    probs = preds[sorted_args][:, 0]
    ids = compare_ids[sorted_args]
    train_labels = real_labels[sorted_args]

    # Remove same id
    cond = np.where(~np.isin(ids, [id]))
    probs = probs[cond]
    ids = ids[cond]
    train_labels = train_labels[cond]

    if verbose:
        print(probs[:print_num])
        print(ids[:print_num])
        print(train_labels[:print_num])
        train_names = []
        for t in train_labels[:10]:
            train_names.append(CLASSES[t])
        print(train_names)
    exit()

    answ = []
    cond = probs > thr
    res = dict()
    prob1 = probs[cond]
    train_labels1 = train_labels[cond]
    for j, p in enumerate(prob1):
        lbl = train_labels1[j]
        if lbl not in res:
            res[lbl] = 0
        res[lbl] += p
    res = sort_dict_by_values(res)
    if verbose:
        print(res)

    for j in range(min(len(res), 5)):
        answ.append(CLASSES[res[j][0]])
    if len(answ) < 5:
        answ.append(CLASSES[0])
    for j, p in enumerate(probs):
        lbl = train_labels[j]
        if CLASSES[lbl] not in answ:
            answ.append(CLASSES[lbl])
        if len(answ) >= 5:
            break
    if verbose:
        print(str(answ))

    score = apk([real_answ], answ, k=5)
    return score, answ


def get_trained_model(fold):
    model, branch_model, head_model = build_model(64e-5, 0, img_shape=(BOX_SIZE, BOX_SIZE, 3))
    dir_path = MODELS_PATH + 'Res_v6_seresnext50/'

    # Get best model from previous run
    best_models = glob.glob(dir_path + 'ft_v6_384px_finetune_{}_*.model'.format(fold))
    best_score = -1.0
    best_weights = ''
    for m in best_models:
        score = float(m[:-6].split('_')[-1])
        if score > best_score:
            best_score = score
            best_weights = m

    print('Best score on previous run: {}. Use weights: {}'.format(best_score, best_weights))
    weights = best_weights
    model.load_weights(weights)
    return model, branch_model, head_model


def proc_all_images(fold):
    score_type = 'max'
    CLASSES = get_classes_array()
    train_part, valid_part = get_kfold_split_weimin(fold)
    print('Train: {} Valid: {}'.format(len(train_part), len(valid_part)))

    model, branch_model, head_model = get_trained_model(fold)

    train_res, train_ids, labels = process_train_images_with_branch_model(branch_model, fold)
    test_res, test_ids = process_tst_images_with_branch_model(branch_model, fold)

    # Find IDs of train and validation in train_res
    valid_indexes = []
    train_ids_list = list(train_ids)
    for i in range(len(valid_part)):
        index1 = train_ids_list.index(valid_part[i])
        valid_indexes.append(index1)

    train_indexes = []
    train_ids_list = list(train_ids)
    for i in range(len(train_part)):
        index1 = train_ids_list.index(train_part[i])
        train_indexes.append(index1)


    if 1:
        # validation vs full train part

        # Save IDs and Real Labels
        compare_ids_valid = []
        real_labels_valid = []
        for i in range(len(valid_part)):
            index1 = valid_indexes[i]
            compare_ids_valid.append(train_ids[index1])
            real_labels_valid.append(labels[train_ids[index1]])

        compare_ids_train = []
        real_labels_train = []
        for j in range(len(train_ids)):
            if labels[train_ids[j]] == 0:
                continue
            compare_ids_train.append(train_ids[j])
            real_labels_train.append(labels[train_ids[j]])

        save_in_file((compare_ids_valid, real_labels_valid, compare_ids_train, real_labels_train),
                     CACHE_PATH + 'ids_fold_{}_valid_vs_full_train.pklz'.format(fold))

        score_sum = 0.0
        scores = []
        # Get predictions
        for i in range(len(valid_part)):
            index1 = valid_indexes[i]
            real_valid_answ = CLASSES[labels[train_ids[index1]]]
            print('Go for valid vs full train {} [{}] Real answ: {}'.format(valid_part[i], i, real_valid_answ))
            cache_path = CACHE_PATH + 'valid_vs_full_train_{}.pklz'.format(valid_part[i])
            if not os.path.isfile(cache_path):
                element1 = []
                element2 = []
                for j in range(len(train_ids)):
                    if labels[train_ids[j]] == 0:
                        continue
                    # print('Go for train {} [Label: {} {}]'.format(train_ids[j], labels[j], CLASSES[labels[j]]))
                    element1.append(train_res[index1])
                    element2.append(train_res[j])
                element1 = np.array(element1)
                element2 = np.array(element2)
                # print(element1.shape, element2.shape, compare_ids, real_labels)
                preds = head_model.predict([element1, element2], batch_size=32)
                preds = np.round(preds, 6)
                save_in_file(preds, cache_path)
            else:
                preds = load_from_file(cache_path)

            # Set same id result as 0
            cond = np.where(np.isin(compare_ids_train, [valid_part[i]]))[0]
            preds[cond, :] = 0

            if score_type == 'avg':
                score, answ_gen = get_single_score(preds, real_valid_answ, compare_ids_train, real_labels_train, CLASSES, 0.99)
            else:
                score, answ_gen = get_single_score_max(preds, real_valid_answ, compare_ids_train, real_labels_train,
                                                   CLASSES, 0.992)
            score_sum += score
            scores.append(score)
            print('Score: {} Avg score: {} Pred answ: {}'.format(score, score_sum / (i + 1), answ_gen))
        scores = np.array(scores)
        print('THR: {} Overall MAP5: {:.6f}'.format(0.99, scores.mean()))

    if 1:
        # Save IDs and Real Labels
        compare_ids = []
        real_labels = []
        for j in range(len(train_ids)):
            if labels[train_ids[j]] == 0:
                continue
            # print('Go for train {} [Label: {} {}]'.format(train_ids[j], labels[j], CLASSES[labels[j]]))
            compare_ids.append(train_ids[j])
            real_labels.append(labels[train_ids[j]])
        save_in_file((compare_ids, real_labels), CACHE_PATH + 'ids_fold_{}_test_vs_train.pklz'.format(fold))

        # Test vs train part
        for i in range(len(test_ids)):
            print('Go for test {} [{}]'.format(test_ids[i], i))
            element1 = []
            element2 = []
            for j in range(len(train_ids)):
                if labels[train_ids[j]] == 0:
                    continue
                # print('Go for train {} [Label: {} {}]'.format(train_ids[j], labels[j], CLASSES[labels[j]]))
                element1.append(test_res[i])
                element2.append(train_res[j])
            element1 = np.array(element1)
            element2 = np.array(element2)
            # print(element1.shape, element2.shape, compare_ids, real_labels)
            preds = head_model.predict([element1, element2], batch_size=32)
            preds = np.round(preds, 6)
            # save_in_file((preds, compare_ids, real_labels), CACHE_PATH + 'test_{}_fold_{}.pklz'.format(test_ids[i], fold))
            save_in_file(preds, CACHE_PATH + 'test_{}_fold_{}.pklz'.format(test_ids[i], fold))


def create_csv_files_tst(fold):
    print('Create csv file for test fold: {}'.format(fold))
    test_df = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    test_ids = test_df['Image'].values
    out = open(CACHE_PATH + 'full_test_vs_train_dict_fold_{}.csv'.format(fold), 'w')
    out.write('ID1,ID2,prob\n')
    compare_ids, real_labels = load_from_file(CACHE_PATH + 'ids_fold_{}_test_vs_train.pklz'.format(fold))
    for i, id in enumerate(test_ids):
        print('Go for ID: {} [{}]'.format(id, i))
        preds = load_from_file(CACHE_PATH + 'test_{}_fold_{}.pklz'.format(id, fold))
        for j, id_train in enumerate(compare_ids):
            out.write("{},{},{:.8f}\n".format(id[:-4], id_train[:-4], preds[j, 0]))
    out.close()


def create_single_matrix_valid(fold):
    print('Create valid matrix for fold: {}'.format(fold))
    preds_full = []
    compare_ids_valid, real_labels_valid, compare_ids_train, real_labels_train = \
        load_from_file(CACHE_PATH + 'ids_fold_{}_valid_vs_full_train.pklz'.format(fold))
    for i, id in enumerate(compare_ids_valid):
        print('Go for ID: {} [{}]'.format(id, i))
        preds = load_from_file(CACHE_PATH + 'valid_vs_full_train_{}.pklz'.format(id))
        preds_full.append(preds[:, 0])
    preds_full = np.array(preds_full)
    print(preds_full.shape)
    save_in_file_fast((compare_ids_valid, compare_ids_train, preds_full), CACHE_PATH + 'full_valid_vs_full_train_matrix_fold_{}.pkl'.format(fold))


def create_single_matrix_tst(fold):
    print('Create test matrix for fold: {}'.format(fold))
    test_df = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    test_ids = test_df['Image'].values

    preds_full = []
    compare_ids, real_labels = load_from_file(CACHE_PATH + 'ids_fold_{}_test_vs_train.pklz'.format(fold))
    for i, id in enumerate(test_ids):
        # print('Go for ID: {} [{}]'.format(id, i))
        preds = load_from_file(CACHE_PATH + 'test_{}_fold_{}.pklz'.format(id, fold))
        preds_full.append(preds[:, 0])
    preds_full = np.array(preds_full)
    print(preds_full.shape) # (7960, 18871)
    save_in_file_fast((test_ids, compare_ids, preds_full), CACHE_PATH + 'full_test_vs_train_matrix_fold_{}.pkl'.format(fold))


def create_averaged_tst_matrix():
    preds_overall = []
    test_df = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    test_ids = test_df['Image'].values
    for fold in range(4):
        print('Create test matrix for fold: {}'.format(fold))
        preds_full = []
        compare_ids, real_labels = load_from_file(CACHE_PATH + 'ids_fold_{}_test_vs_train.pklz'.format(fold))
        for i, id in enumerate(test_ids):
            # print('Go for ID: {} [{}]'.format(id, i))
            preds = load_from_file(CACHE_PATH + 'test_{}_fold_{}.pklz'.format(id, fold))
            preds_full.append(preds[:, 0])
        preds_full = np.array(preds_full)
        print(preds_full.shape) # (7960, 18871)
        preds_overall.append(preds_full)
    preds_overall = np.array(preds_overall).mean(axis=0)
    print(preds_overall.shape)
    save_in_file_fast((test_ids, compare_ids, preds_overall), CACHE_PATH + 'full_test_vs_train_matrix_avg.pkl')


if __name__ == '__main__':
    start_time = time.time()
    proc_all_images(0)
    proc_all_images(1)
    proc_all_images(2)
    proc_all_images(3)
    if 1:
        for i in range(4):
            create_single_matrix_valid(i)
        create_averaged_tst_matrix()
    print('Time: {:.0f} sec'.format(time.time() - start_time))

'''
Full matrix
Fold 0: THR: 0.99 Overall MAP5: 0.964152 (Max: 0.963637)
Fold 1: THR: 0.99 Overall MAP5: 0.963566 (Max: 0.963064)
Fold 2: THR: 0.99 Overall MAP5: 0.962459 (Max: 0.961415)
Fold 3: THR: 0.99 Overall MAP5: 0.963878 (Max: 0.964524)
'''