# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
from multiprocessing.pool import ThreadPool, Pool
from functools import partial


CACHE_PATH = OUTPUT_PATH + 'seamese_net_v5_rgb_densenet121_512px/'
NUM_FOLDS = 4
CLASSES = get_classes_array()


def get_answer(id, thr, compare_ids, real_labels):
    print('Go for {}'.format(id))

    preds_list = []
    for fold in range(NUM_FOLDS):
        preds = load_from_file(CACHE_PATH + 'test_{}_fold_{}.pklz'.format(id, fold))
        preds_list.append(preds)
    preds = np.array(preds_list)

    preds = preds.mean(axis=0)
    sorted_args = np.argsort(preds[:, 0])[::-1]

    probs = preds[sorted_args][:, 0]
    train_ids = compare_ids[sorted_args]
    train_labels = real_labels[sorted_args]
    if 0:
        print(probs[:10])
        print(train_ids[:10])
        print(train_labels[:10])
    train_names = []
    for t in train_labels[:10]:
        train_names.append(CLASSES[t])
    if 0:
        print(train_names)

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
    if 0:
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
    if 0:
        print(str(answ) + '\n\n')

    if 0:
        test_image = read_single_image(INPUT_PATH + 'test/' + id)
        show_image(test_image)
        for t in train_ids:
            train_image = read_single_image(INPUT_PATH + 'train/' + t)
            show_image(train_image)

    return answ


def create_submission(thr):
    out_path = SUBM_PATH + 'siamese_net_v5_densenet121_512px_THR_{}.csv'.format(thr)
    out = open(out_path, 'w')
    out.write('Image,Id\n')
    test_df = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    test_ids = test_df['Image'].values

    for fold in range(4):
        compare_ids, real_labels = load_from_file(CACHE_PATH + 'ids_fold_{}_test_vs_train.pklz'.format(fold))
        compare_ids = np.array(compare_ids)
        real_labels = np.array(real_labels)

    p = Pool(8)
    process_item_func = partial(get_answer, thr=thr, compare_ids=compare_ids, real_labels=real_labels)
    answers = p.map(process_item_func, list(test_ids))

    for i, id in enumerate(test_ids):
        print('Go for {} [{}]'.format(id, i))
        answ = answers[i]

        out.write(test_ids[i] + ',')
        out.write(' '.join(answ))
        out.write('\n')

    out.close()
    # compare_submissions(SUBM_PATH + 'Ensemble_seven_model_v2.0_threshold_0.51_LB_0.952.csv', out_path)
    check_submission_distribution(out_path)


if __name__ == '__main__':
    start_time = time.time()
    create_submission(thr=0.85)
    print('Time: {:.0f} sec'.format(time.time() - start_time))


'''
THR 0.70: Unique: 3982 'new_whale', 2189
THR 0.75: Unique: 3962 'new_whale', 2244
THR 0.80: Unique: 3943 'new_whale', 2291 LB: 0.959
THR 0.85: Unique: 3925 'new_whale', 2322
'''