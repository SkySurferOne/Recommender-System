import copy
import random


def estimate(data_arr, resd, ave_data_arr, user_mean, item_mean, pcnt):
    users_num = len(data_arr)
    item_num = len(data_arr[0])
    estm_arr = copy.deepcopy(data_arr)
    item_estm_list = [0] * item_num

    for j in range(item_num):
        user_count = 0.0

        for i in range(users_num):
            if data_arr[i][j] > 0:
                item_estm_list[j] += ave_data_arr[i][j]
            user_count += 1

        if user_count == 0:
            user_count += 1

        item_estm_list[j] /= user_count
    for i in range(users_num):
        for j in range(item_num):
            if estm_arr[i][j] == 0:
                ran = random.random()
                if ran > pcnt:
                    continue
            estm_arr[i][j] = user_mean[i] + item_estm_list[j]

            if estm_arr[i][j] < 0:
                estm_arr[i][j] = 1
            elif estm_arr[i][j] > 5:
                estm_arr[i][j] = 5

    return estm_arr


def estimate_new(data_arr, cluster_res, ave_data_arr, user_mean, item_mean, pcnt):
    """
    This function stores the pre-calculated user and item similarity and mean in array.

    :param data_arr:
    :param cluster_res: user assignments to cluster
    :param ave_data_arr:
    :param user_mean:
    :param item_mean:
    :param pcnt:
    :return:
    """
    user_num = len(data_arr)
    item_num = len(data_arr[0])
    estm_arr = copy.deepcopy(data_arr)

    # ============
    user_cluster_mates = []
    for usr_i in range(user_num):
        users_in_same_cluster = set()

        # get users which are in the similar cluster as i
        for usr_j in range(user_num):
            if usr_i != usr_j and cluster_res[usr_i] == cluster_res[usr_j]:
                users_in_same_cluster.add(usr_j)
        user_cluster_mates.append(users_in_same_cluster)

    # ============
    similar_items = []
    for item_i in range(item_num):
        items_with_similar_avg_rating = set()

        # get items which have similar rating to i
        for item_j in range(item_num):
            if item_i != item_j and abs(item_mean[item_i] - item_mean[item_j]) < 0.1:
                items_with_similar_avg_rating.add(item_j)
        similar_items.append(items_with_similar_avg_rating)

    # ============
    for item_i in range(item_num):
        for usr_i in range(user_num):
            # if user didnt rated set him a mean
            if estm_arr[usr_i][item_i] == 0:
                estm_arr[usr_i][item_i] = user_mean[usr_i]

            user_v = 0
            user_count = 0
            for usr_i_mate in user_cluster_mates[usr_i]:
                if data_arr[usr_i_mate][item_i] > 0:
                    user_v += ave_data_arr[usr_i_mate][item_i]
                    user_count += 1

                if user_count == 0:
                    user_count += 1
                user_v /= user_count
                estm_arr[usr_i][item_i] += 0.5 * user_v

            item_v = 0
            item_count = 0
            for similar_item_to_i in similar_items[item_i]:
                if data_arr[usr_i][similar_item_to_i] > 0:
                    item_v += ave_data_arr[usr_i][similar_item_to_i]
                    item_count += 1

                if item_count == 0:
                    item_count += 1

                item_v /= item_count
                estm_arr[usr_i][item_i] += 0.5 * item_v

            if estm_arr[usr_i][item_i] < 0:
                estm_arr[usr_i][item_i] = 1
            elif estm_arr[usr_i][item_i] > 5:
                estm_arr[usr_i][item_i] = 5

    return estm_arr
