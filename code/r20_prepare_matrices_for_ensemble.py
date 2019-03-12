# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
from scipy.sparse import csr_matrix


def get_tagged_data():
    # load tagged data
    tagged = dict([(p[:-4] + '.jpg', w) for _, p, w in pd.read_csv(INPUT_PATH + 'train.csv').to_records()])
    # print("Total tagged images: ", len(tagged))
    return tagged


def get_real_answers(ids):
    tagged = get_tagged_data()
    answ = []
    for id in ids:
        try:
            answ.append(tagged[id + '.jpg'])
        except:
            answ.append(tagged[list(tagged.keys())[0]])
    return np.array(answ)


def get_single_score(preds, id, compare_ids, real_answ, real_labels, thr):
    print_num = 5
    verbose = False

    # Remove same id
    cond = np.where(np.isin(compare_ids, [id]))
    preds[cond] = -1

    sorted_args = np.argsort(preds)[::-1]
    probs = preds[sorted_args]
    ids = compare_ids[sorted_args]
    train_labels = real_labels[sorted_args]

    if verbose:
        print(real_answ)
        print(probs[:print_num])
        print(ids[:print_num])
        print(train_labels[:print_num])
        train_names = []
        for t in train_labels[:10]:
            train_names.append(t)
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
    if verbose:
        print(res)

    for j in range(min(len(res), 5)):
        answ.append(res[j][0])
    if len(answ) < 5:
        answ.append('new_whale')
    for j, p in enumerate(probs):
        lbl = train_labels[j]
        if lbl not in answ:
            answ.append(lbl)
        if len(answ) >= 5:
            break
    if verbose:
        print(str(answ))

    score = apk([real_answ], answ, k=5)
    return score, answ


def get_score_from_matrix(m, base_order_valid, base_order_train, answ_valid, answ_train, thr):
    new_whales_count = 0
    scores = []
    for i in range(m.shape[0]):
        preds = m[i].copy()
        score, answ1 = get_single_score(preds, base_order_valid[i], base_order_train, answ_valid[i], answ_train, thr)
        # print('ID: {} Score: {} Avg score: {} Real answ: {} Answ: {}'.format(ids_valid[i], score, score_sum / (i + 1), answ_valid[i], answ1))
        if answ1[0] == 'new_whale':
            new_whales_count += 1
        scores.append(score)
    return scores, new_whales_count


def concat_fold_matrix(matrix_arr, base_order_valid, base_order_train, answ_valid, answ_train):
    matrix_arr_full = []
    base_order_valid = np.concatenate(base_order_valid)
    # check if base_order train is the same
    for i in range(4):
        for j in range(i+1, 4):
            if tuple(base_order_train[i]) != tuple(base_order_train[j]):
                print('Error base_order_train!')
                exit()
            if tuple(answ_train[i]) != tuple(answ_train[j]):
                print('Error answ_train!')
                exit()
    base_order_train = base_order_train[0]
    answ_valid = np.concatenate(answ_valid)
    answ_train = answ_train[0]
    m = []
    for fold in range(4):
        part = np.array(matrix_arr[fold])
        m.append(part)
    m = np.concatenate(m)
    matrix_arr_full.append(m)
    matrix_arr = np.array(matrix_arr_full)
    print(matrix_arr.shape)
    print(base_order_valid.shape)
    print(base_order_train.shape)
    print(answ_valid.shape)
    print(answ_train.shape)
    return matrix_arr, base_order_valid, base_order_train, answ_valid, answ_train


def zerofy_same_ids(matrix_arr, base_order_valid, base_order_train):
    for i in range(matrix_arr.shape[1]):
        # Remove same id
        cond = np.where(np.isin(base_order_train, [base_order_valid[i]]))
        matrix_arr[:, i, cond] = 0
    return matrix_arr


def remove_tst_data(matrix_arr, base_order_train, answ_train):
    s = list(pd.read_csv(INPUT_PATH + 'sample_submission.csv')['Image'].values)

    print('Initial matrix shape: {}'.format(matrix_arr.shape))
    use_ids = []
    total = 0
    for i in range(len(base_order_train)):
        if base_order_train[i] + '.jpg' not in s:
            use_ids.append(i)
        else:
            total += 1
    use_ids = np.array(use_ids)
    print('Removed IDs: {}'.format(total))

    matrix_arr = matrix_arr[:, :, use_ids]
    base_order_train = base_order_train[use_ids]
    answ_train = answ_train[use_ids]
    print('Updated matrix shape: {}'.format(matrix_arr.shape))
    return matrix_arr, base_order_train, answ_train


def create_matrix_train(tables_path, out_path):
    EPS = 0.00001
    print('Go for: {}'.format(tables_path))
    matrix_arr = []
    base_order_valid = []
    base_order_train = []
    answ_valid = []
    answ_train = []
    scores_full = []
    for fold in range(4):
        matrix_arr.append([])
        base_order_valid.append(None)
        base_order_train.append(None)
        answ_valid.append([])
        answ_train.append([])

    for fold in range(4):
        file = tables_path
        file = file.replace('*', str(fold))
        print('Read {}'.format(file))
        data = load_from_file_fast(file)
        order_valid = np.array(data[0])
        order_train = np.array(data[1])
        m = data[2]

        order_valid = np.array([x[:-4] for x in order_valid])
        order_train = np.array([x[:-4] for x in order_train])

        asort_valid = np.argsort(order_valid)
        asort_train = np.argsort(order_train)
        print(m.shape)
        m = m[asort_valid, :]
        m = m[:, asort_train]

        base_order_valid[fold] = order_valid[asort_valid]
        base_order_train[fold] = order_train[asort_train]
        answ_valid[fold] = get_real_answers(base_order_valid[fold])
        answ_train[fold] = get_real_answers(base_order_train[fold])

        scores, nw = get_score_from_matrix(m, base_order_valid[fold], base_order_train[fold], answ_valid[fold], answ_train[fold], thr=0.99)
        scores_full += scores
        matrix_arr[fold] = m
        print('Fold score: {:.6f}'.format(np.array(scores).mean()))

    print('Overall score: {:.6f}'.format(np.array(scores_full).mean()))
    # Concatenate all folds
    matrix_arr, base_order_valid, base_order_train, answ_valid, answ_train = \
        concat_fold_matrix(matrix_arr, base_order_valid, base_order_train, answ_valid, answ_train)

    # Remove pseudolabel test data
    matrix_arr, base_order_train, answ_train = remove_tst_data(matrix_arr, base_order_train, answ_train)

    # Set 0 to same IDs
    matrix_arr = zerofy_same_ids(matrix_arr, base_order_valid, base_order_train)

    # Prepare final dict
    base_order_valid = np.array([x + '.jpg' for x in base_order_valid])
    base_order_train = np.array([x + '.jpg' for x in base_order_train])

    print(base_order_valid)
    print(base_order_train)

    # Prepare sparse matrix
    matrix_arr = matrix_arr[0]
    matrix_arr[matrix_arr < EPS] = 0.0
    matrix_arr = csr_matrix(matrix_arr)
    print(matrix_arr.shape)

    out = dict()
    out['row_names'] = base_order_valid
    out['col_names'] = base_order_train
    out['overall_score'] = np.array(scores_full).mean()
    out['val_vs_train_mat_sparse'] = matrix_arr

    # Save to file
    save_in_file_fast(out, out_path)


def create_matrix_tst(tables_path, out_path):
    EPS = 0.00001

    out_path = out_path[:-4] + '-test.pkl'

    matrix_arr = []
    file = tables_path
    print('Read {}'.format(file))
    data = load_from_file_fast(file)

    order_valid = np.array(data[0])
    order_train = np.array(data[1])
    m = data[2]

    # order_valid = np.array([x[:-4] for x in order_valid])
    # order_train = np.array([x[:-4] for x in order_train])

    asort_valid = np.argsort(order_valid)
    asort_train = np.argsort(order_train)
    print(m.shape)
    m = m[asort_valid, :]
    m = m[:, asort_train]

    base_order_valid = order_valid[asort_valid]
    base_order_train = order_train[asort_train]

    matrix_arr.append(m.copy())

    matrix_arr = np.array(matrix_arr)
    print(matrix_arr.shape)

    intersection = (set(base_order_valid) & set(base_order_train))
    print('Intersection of IDs:', len(intersection))

    # Set 0 to same IDs
    matrix_arr = zerofy_same_ids(matrix_arr, base_order_valid, base_order_train)

    # Prepare sparse matrix
    matrix_arr = matrix_arr[0]
    matrix_arr[matrix_arr < EPS] = 0.0
    matrix_arr = csr_matrix(matrix_arr)
    print(matrix_arr.shape)

    out = dict()
    out['row_names'] = base_order_valid
    out['col_names'] = base_order_train
    # name = os.path.basename(out_path).split('-')[2]
    out['test_vs_train_mat_sparse'] = matrix_arr

    # Save to file
    save_in_file_fast(out, out_path)


def get_matrix_list():
    matrix_list = [
        [
            # Overall score: 0.964212
            OUTPUT_PATH + 'seamese_net_v5_rgb_densenet121_512px/full_valid_vs_full_train_matrix_fold_*.pkl',
            OUTPUT_PATH + 'seamese_net_v5_rgb_densenet121_512px/full_test_vs_train_matrix_avg.pkl',
            FEATURES_PATH + 'cv-analysis-fs14-LB959-densenet121-512px-sparse.pkl',
        ],
        [
            # Overall score: 0.963513
            OUTPUT_PATH + 'seamese_net_v6_rgb_seresnext50_384px/full_valid_vs_full_train_matrix_fold_*.pkl',
            OUTPUT_PATH + 'seamese_net_v6_rgb_seresnext50_384px/full_test_vs_train_matrix_avg.pkl',
            FEATURES_PATH + 'cv-analysis-fs16-LB959-seresnext50-384px-sparse.pkl',
        ],
    ]
    return matrix_list


if __name__ == '__main__':
    matrix_list = get_matrix_list()
    start_time = time.time()
    for m in matrix_list:
        create_matrix_train(m[0], m[2])
    for m in matrix_list:
        create_matrix_tst(m[1], m[2])
    print('Time: {:.0f} sec'.format(time.time() - start_time))
