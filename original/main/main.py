import copy
import math
import sys

from original.main import cluster
from original.main import estimation
from original.main import evaluate
from original.main import res_pic


def init_arr(r, c):
    """
    init array with 0
    """
    arr = []
    for i in range(r):
        tmp_arr = [0] * c
        arr.append(tmp_arr)

    return arr


def init_arr_n(r, c):
    """
    init array with n -1
    """
    arr = []
    for i in range(r):
        tmp_arr = [-1] * c
        arr.append(tmp_arr)

    return arr


def get_input(filename, item_num, user_num):
    """
    get input data
    """
    data_arr = init_arr(user_num, item_num)
    with open(filename) as f:
        for lines in f:
            arr = lines.strip().split('\t')

            u = int(arr[0]) - 1
            i = int(arr[1]) - 1
            r = int(arr[2])
            data_arr[u][i] = r

    return data_arr


def get_ave_data_arr(data_arr, user_mean):
    """
    get average rank ( data - user_mean)
    """
    user_num = len(data_arr)
    item_num = len(data_arr[0])
    ave_data_arr = copy.deepcopy(data_arr)

    for i in range(user_num):
        for j in range(item_num):
            ave_data_arr[i][j] -= user_mean[i]

    return ave_data_arr


def get_pear_data_arr(data_arr, item_mean):
    """
    get average rank array ( data - item_mean):
    used for pearson similarity calculation
    """
    user_num = len(data_arr)
    item_num = len(data_arr[0])
    pear_data_arr = copy.deepcopy(data_arr)

    for i in range(user_num):
        for j in range(item_num):
            pear_data_arr[i][j] -= item_mean[j]

    return pear_data_arr


def get_sqr_data_arr(ave_data_arr):
    """
    get square of data
    """
    user_num = len(ave_data_arr)
    item_num = len(ave_data_arr[0])
    sqr_data_arr = copy.deepcopy(ave_data_arr)
    sqr_data_list = [0.0] * item_num
    for i in range(user_num):
        for j in range(item_num):
            sqr_data_arr[i][j] = ave_data_arr[i][j] * ave_data_arr[i][j]

    for l in range(item_num):
        sq_sum = 0
        for m in range(user_num):
            sq_sum += sqr_data_arr[m][l]
            sqr_data_list[l] = math.sqrt(sq_sum)

        if sqr_data_list[l] == 0:
            sqr_data_list[l] = 1

    return sqr_data_list


def get_user_mean(data_arr):
    """
    get user mean
    """
    user_num = len(data_arr)
    item_num = len(data_arr[0])
    user_mean = [0.0] * user_num
    for i in range(user_num):
        user_ave = 0.0
        user_count = 0.0
        for j in range(item_num):
            if data_arr[i][j] > 0:
                user_ave += data_arr[i][j]
                user_count += 1

        if user_count > 0:
            user_mean[i] = user_ave / item_num
        else:
            user_mean[i] = 0

    return user_mean


def get_item_mean(data_arr):
    """
    get item mean
    """
    user_num = len(data_arr)
    item_num = len(data_arr[0])
    item_mean = [0.0] * item_num

    for j in range(item_num):
        user_ave = 0.0
        user_count = 0.0

        for i in range(user_num):
            if data_arr[i][j] > 0:
                user_ave += data_arr[i][j]
                user_count += 1

        if user_count > 0:
            item_mean[j] = user_ave / user_num
        else:
            item_mean[j] = 0

    return item_mean


def cosin_similarity(ave_data_arr, sqr_data_list, data_arr):
    """
    calculate cosin similarity
    """
    user_num = len(ave_data_arr)
    item_num = len(ave_data_arr[0])
    sim_arr = init_arr(item_num, item_num)

    for i in range(item_num):
        for j in range(0, i):
            sim_arr[i][j] = sim_arr[j][i]

        sim_arr[i][i] = 1

        for j in range(i + 1, item_num):
            ms = 0.0
            for l in range(user_num):
                sim_arr[i][j] += ave_data_arr[l][i] * ave_data_arr[l][j]
                sim_arr[i][j] /= (sqr_data_list[i] * sqr_data_list[j])


    return sim_arr


def sim_sort(sim_arr):
    """
    sorted by similarity for each item
    """
    item_num = len(sim_arr)
    sim_sorted = []
    for i in range(item_num):
        temp = {j: sim_arr[i][j] for j in range(0, len(sim_arr[i]))}
        sim_s = sorted(temp.items(), key=lambda k: k[1], reverse=True)
        tmp_list = [sim_s[j][0] for j in range(0, len(sim_arr))]
        sim_sorted.append(tmp_list)

    return sim_sorted


def rec(data_arr, sim_sorted, sim_arr, top_rec):
    """
    calculate the recommand result
    """
    res_arr = copy.deepcopy(data_arr)
    user_num = len(data_arr)
    item_num = len(data_arr[0])

    for i in range(user_num):
        for j in range(item_num):
            user_count = 0
            if res_arr[i][j] == 0:
                user_count = 0

            sim_sum = 0.0
            for k in sim_sorted[j]:
                if k == j:
                    continue

                if user_count >= top_rec:
                    break

                if res_arr[i][k] > 0:
                    user_count += 1

                res_arr[i][j] += sim_arr[j][k] * data_arr[i][k]
                sim_sum += abs(sim_arr[j][k])

                if sim_sum == 0:
                    sim_sum = 1
                res_arr[i][j] /= sim_sum
                if res_arr[i][j] < 1:
                    # print " " + str(res_arr[i][j])
                    res_arr[i][j] = 1
                elif res_arr[i][j] > 5:
                    # print " " + str(res_arr[i][j])
                    res_arr[i] = 5

    return res_arr


def get_rec_item(data_arr, res_arr, k_rec):
    """
    get the k_rec items for recommendation
    """
    user_num = len(data_arr)
    item_num = len(data_arr[0])
    rec_items = init_arr_n(user_num, k_rec)

    for i in range(user_num):
        tmp_list = []

        for j in range(item_num):
            if data_arr[i][j] == 0 and res_arr[i][j] > 0:
                tmp_list.append([j, res_arr[i][j]])

        tmp_sorted_list = sorted(tmp_list, key=lambda k: k[1], reverse=True)
        tmp_sorted_list = tmp_sorted_list[:k_rec]

        for m in range(len(tmp_sorted_list)):
            rec_items[i][m] = tmp_sorted_list[m][0]

    return rec_items


def main():
    """
    single round calculation
    """
    in_file = sys.argv[1]
    test_file = sys.argv[2]
    # out_file = sys.argv[3]
    # user_num = 5
    # item_num = 8
    user_num = 943
    item_num = 1682
    data_arr = get_input(in_file, item_num, user_num)
    test_arr = get_input(test_file, item_num, user_num)
    user_mean = get_user_mean(data_arr)
    item_mean = get_item_mean(data_arr)

    cluster_num = 1
    k_neighbour = 40
    # k_neighbour = int(sys.argv[3])
    k_rec = 10
    # pcnt = float(sys.argv[3])
    pcnt = 1

    # clusterization
    cluster_res = cluster.kmean(data_arr, cluster_num)

    # calc similarities
    ave_data_arr = get_ave_data_arr(data_arr, user_mean)
    estm_arr = estimation.estimate_new(data_arr,
                                       cluster_res,
                                       ave_data_arr,
                                       user_mean,
                                       item_mean,
                                       pcnt)

    ave_data_arr = get_ave_data_arr(data_arr, user_mean)
    sqr_data_list = get_sqr_data_arr(ave_data_arr)
    ave_data_arr = get_ave_data_arr(estm_arr, user_mean)
    sqr_data_list = get_sqr_data_arr(ave_data_arr)

    # pear_data_arr = get_pear_data_arr(estm_arr, item_mean)
    # pear_sqr_data = get_sqr_data_arr(pear_data_arr)

    sim_arr = cosin_similarity(ave_data_arr, sqr_data_list, estm_arr)

    # sim_arr = cosin_similarity(pear_data_arr, pear_sqr_data, estm_arr)
    # sim_arr = cosin_similarity(pear_data_arr, pear_sqr_data, data_arr)
    # sim_arr = cosin_similarity(ave_data_arr, sqr_data_list, data_arr)

    sim_sorted = sim_sort(sim_arr)

    # calculate recommendations
    # res_arr = rec(data_arr, sim_sorted, sim_arr, k_neighbour)
    res_arr = rec(estm_arr, sim_sorted, sim_arr, k_neighbour)
    rec_items = get_rec_item(data_arr, res_arr, k_rec)

    rmse = evaluate.rmse(test_arr, res_arr)
    print('rmse: %.5f' % rmse)
    mae = evaluate.mae(test_arr, res_arr)
    print('mae: %.5f' % mae)

    pres, recall, f1 = evaluate.pre_recal(test_arr, rec_items)
    print('precise: %.5f' % pres)
    print('recall: %.5f' % recall)
    print('f1: %.5f' % f1)


def run(cluster_num, k_neighbour, k_rec, pcnt, data_arr, test_arr, user_mean, item_mean):
    """
    run the process
    """
    cluster_res = None  # ???
    ave_data_arr = get_ave_data_arr(data_arr, user_mean)
    estm_arr = estimation.estimate_new(data_arr, cluster_res,
                                       ave_data_arr, user_mean,
                                       item_mean, pcnt)
    ave_data_arr = get_ave_data_arr(data_arr, user_mean)
    sqr_data_list = get_sqr_data_arr(ave_data_arr)

    ave_data_arr = get_ave_data_arr(estm_arr, user_mean)
    sqr_data_list = get_sqr_data_arr(ave_data_arr)

    # pear_data_arr = get_pear_data_arr(estm_arr, item_mean)
    # pear_sqr_data = get_sqr_data_arr(pear_data_arr)

    sim_arr = cosin_similarity(ave_data_arr, sqr_data_list, estm_arr)

    # sim_arr = cosin_similarity(pear_data_arr, pear_sqr_data, estm_arr)
    # sim_arr = cosin_similarity(pear_data_arr, pear_sqr_data, data_arr)
    # sim_arr = cosin_similarity(ave_data_arr, sqr_data_list, data_arr)

    sim_sorted = sim_sort(sim_arr)

    # res_arr = rec(data_arr, sim_sorted, sim_arr, k_neighbour)

    res_arr = rec(estm_arr, sim_sorted, sim_arr, k_neighbour)
    rec_items = get_rec_item(data_arr, res_arr, k_rec)

    # rmse = evaluate.rmse(test_arr, res_arr)
    # mae = evaluate.mae(test_arr, res_arr)
    # pres, recall, f1 = evaluate.pre_recal(test_arr, rec_items)

    rmse = evaluate.rmse(test_arr, res_arr)
    print('rmse: %.5f' % rmse)

    mae = evaluate.mae(test_arr, res_arr)
    print('mae: %.5f' % mae)

    pres, recall, f1 = evaluate.pre_recal(test_arr, rec_items)
    print('precise: %.5f' % pres)
    print('recall: %.5f' % recall)
    print('f1: %.5f' % f1)

    return rmse, mae, pres, recall, f1


def test():
    in_file = sys.argv[1]
    u = 6
    i = 8
    data_arr = get_input(in_file, i, u)
    test_arr = get_input(in_file, i, u)
    # data_arr = init_arr(3, 4)
    # data_arr = init_arr_n(3, 4)

    print('data:')
    print(data_arr)

    user_mean = get_user_mean(data_arr)
    print('user_mean:')
    print(user_mean)

    item_mean = get_item_mean(data_arr)
    print('item_mean:')
    print(item_mean)

    ave_data_arr = get_ave_data_arr(data_arr, user_mean)
    print('ave_data_arr:')
    print(ave_data_arr)

    pear_data_arr = get_pear_data_arr(data_arr, item_mean)
    print('pear_ave_data_arr:')
    print(pear_data_arr)

    sqr_data_arr = get_sqr_data_arr(ave_data_arr)
    print('sqr_data_arr:')
    print(sqr_data_arr)

    sim_arr = cosin_similarity(ave_data_arr, sqr_data_arr, data_arr)
    print('sim_arr:')
    print(sim_arr)

    sim_arr_sorted = sim_sort(sim_arr)
    print('sim_sorted:')
    print(sim_arr_sorted)

    res = rec(data_arr, sim_arr_sorted, sim_arr, 2)
    print('res arr:')
    print(res)

    print(test_arr)
    rmse = evaluate.rmse(test_arr, res)
    print('rmse: %.5f' % rmse)

    mae = evaluate.mae(test_arr, res)
    print('mae: %.5f' % mae)
    # pres, recall, f1 = evaluate.pre_recal(test_arr, rec_items)

    pic_file = './ad'
    axis_list = [1, 2, 3, 4, 5, 6, 7, 8]
    res_pic.draw(pic_file + '.rmse.png', axis_list, item_mean, 'pcnt', 'rmse')
    print(axis_list)

    res_pic.draw(pic_file + '.f1.png', axis_list, test_arr[0], 'pcnt', 'f1')
    print(axis_list)

    # print user_mean
    # print item_mean

    data_arr[2][3] = 1


# print data_arr
def process():
    in_file = sys.argv[1]
    test_infile = sys.argv[2]
    pic_file = sys.argv[3]
    user_num = 943
    item_num = 1682
    data_arr = get_input(in_file, item_num, user_num)
    test_arr = get_input(test_infile, item_num, user_num)
    user_mean = get_user_mean(data_arr)
    item_mean = get_item_mean(data_arr)
    cluster_num = 3
    k_neighbour = 40
    k_rec = 50
    pcnt = 0.06
    rmse_list = []
    mae_list = []
    pres_list = []
    recall_list = []
    f1_list = []
    axis_list = []

    for i in range(10):
        k_rec = i * 100 + 100
    rmse, mae, pres, recall, f1 = run(cluster_num, k_neighbour,
                                      k_rec, pcnt, data_arr, test_arr,
                                      user_mean, item_mean)

    rmse_list.append(rmse)
    mae_list.append(mae)
    pres_list.append(pres)
    recall_list.append(recall)
    f1_list.append(f1)

    res_pic.draw(pic_file + '.rmse.png', axis_list, rmse_list, 'List', 'RMSE')
    res_pic.draw(pic_file + '.mae.png', axis_list, mae_list, 'List', 'MAE')
    res_pic.draw(pic_file + '.pres.png', axis_list, pres_list, 'List', 'pres')
    res_pic.draw(pic_file + '.recall.png', axis_list, recall_list, 'List', 'recall')
    res_pic.draw(pic_file + '.f1.png', axis_list, f1_list, 'List', 'f1')


if __name__ == '__main__':
    main()
    # test()
    # process()
