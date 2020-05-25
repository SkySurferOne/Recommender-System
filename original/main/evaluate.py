import math


# This file evaluates the recommendation results using
# MAE, RMSE, Precision, Recall and F1 score.

def rmse(test_arr, res_arr):
    r = len(test_arr)
    c = len(test_arr[0])
    sum_res = 0.0
    count = 0
    test_user = []

    for i in range(r):
        for j in range(c):
            if test_arr[i][j] > 0:
                test_user.append(i)

    for i in test_user:
        for j in range(0, c):
            if test_arr[i][j] > 0:
                sum_res += (res_arr[i][j] - test_arr[i][j]) ** 2
                count += 1

    if count == 0:
        raise Exception('No test data', 'in evaluate.py')

    return math.sqrt(sum_res / count)


def mae(test_arr, res_arr):
    r = len(res_arr)
    c = len(res_arr[0])
    sum_res = 0.0
    count = 0
    test_user = []

    for i in range(r):
        for j in range(c):
            if test_arr[i][j] > 0:
                test_user.append(i)
                break

    for i in test_user:
        for j in range(0, c):
            if test_arr[i][j] > 0:
                sum_res += abs(res_arr[i][j] - test_arr[i][j])
                count += 1

    if count == 0:
        raise Exception('No test data', 'in evaluate.py')

    print(count)

    return sum_res / count


def cal(res_arr, test_arr, outfile):
    res_rmse = rmse(res_arr, test_arr)
    res_mae = mae(res_arr, test_arr)
    outfile.write("exp result is : \nrmse : " +
                  str(res_rmse) +
                  "\nmae : " +
                  str(res_mae) +
                  "\n")

    return res_rmse, res_mae


def pre_recal(test_arr, rec_items):
    pres = 0.0
    recall = 0.0
    f1 = 0.0
    r = len(test_arr)
    c = len(test_arr[0])
    tp = 0
    fp = 0
    tn = 0
    test_user = []

    for i in range(r):
        for j in range(c):
            if test_arr[i][j] > 0:
                test_user.append(i)
                break

    for i in test_user:
        for j in range(len(rec_items[i])):
            if test_arr[i][rec_items[i][j]] > 0:
                tp += 1
            else:
                fp += 1

        for j in range(len(test_arr[i])):
            if test_arr[i][j] > 0 and j not in rec_items[i]:
                tn += 1

    print("tp " + str(tp))
    print("tn " + str(tn))
    print("fp " + str(fp))

    pres = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + tn)

    if pres + recall == 0:
        f1 = 0
    else:
        f1 = 2 * pres * recall / (pres + recall)

    return pres, recall, f1
