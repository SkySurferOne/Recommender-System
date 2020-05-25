import copy
import random
import sys


def init_arr(arr, c):
    for i in range(c):
        tmp_arr = [0] * c
        arr.append(tmp_arr)

    return arr


def kmeans_init(data_arr, clus_k):
    """
    Initialize kmeans centroids using random points
    :param data_arr:
    :param clus_k:
    :return: centers, array which denotes cluster - user assignment
    """
    users_num = len(data_arr)
    items_num = len(data_arr[0])
    center = []
    seed = set()
    resd = [-1] * users_num

    for i in range(clus_k):
        idx = int(random.uniform(0, users_num))
        while idx in seed:
            idx = int(random.uniform(0, users_num))
        seed.add(idx)
        print(idx)
        center.append(copy.deepcopy(data_arr[idx]))

    print(len(center))
    print(center[0][:20])

    return center, resd


def power(data_list):
    res = 0.0
    for it in range(len(data_list)):
        res += data_list[it] * data_list[it]

    return res


def getDistance(elements, center):
    # todo: the i-th calculate
    res = 0.0
    for i in range(len(elements)):
        dif = elements[i] - center[i]
        res += dif * dif

    # res = math.sqrt(res)
    return res


def kmean(data_arr, k):
    """
    This function implements the K-means clustering to group items into k clusters.
    
    :param data_arr:
    :param k:
    :return:
    """
    rows = len(data_arr)
    if rows < 1:
        print("data error, stop cluster")
        return None

    cols = len(data_arr[0])
    try:
        centroids, cluster_assignment = kmeans_init(data_arr, k)
    except:
        return None

    count = 0
    flag = True
    while flag and count < 100:
        flag = False
        print('round :' + str(count))

        # cal dist
        for i in range(0, rows):
            dist = sys.maxsize
            cent_id = 0
            for j in range(0, k):
                dist_item = getDistance(data_arr[i], centroids[j])
                if dist_item < dist:
                    dist = dist_item
                cent_id = j

            if cent_id != cluster_assignment[i]:
                cluster_assignment[i] = cent_id
                flag = True

        # cal new center
        for j in range(0, k):
            cluster_num = 0
            tmp_center = [0] * cols
            for it in range(len(cluster_assignment)):
                if cluster_assignment[it] == j:
                    cluster_num += 1
                for l in range(cols):
                    tmp_center[l] += data_arr[it][l]

            if cluster_num == 0:
                print('xxxx')
                raise Exception

            for l in range(cols):
                tmp_center[l] /= float(cluster_num)

            centroids[j] = tmp_center

        count += 1

    return cluster_assignment
