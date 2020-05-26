import copy
import math

import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

from common.utils.LogTime import LogTime
from common.utils.plotter import draw_plot_point_label_set2
from common.utils.utils import merge_labels_2d
from customized.main.DatasetManager import DatasetManager
from customized.main.Evaluator import Evaluator


def pca_dim_reduction(X, n_out=2):
    pca = PCA(n_components=n_out)

    return pca.fit_transform(X)


def clusterize(user_item, k=3, plot_charts=True):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(user_item)

    if plot_charts:
        user_item_reduced = pca_dim_reduction(user_item, n_out=2)
        original_set = merge_labels_2d(user_item_reduced, kmeans.labels_)
        draw_plot_point_label_set2('kmeans (k={}) - User ratings'.format(k), 'x', 'y', original_set)

    return kmeans.labels_, kmeans


def predict_ksimilar_user(ratings, ratings_diff, similarity, k=20):
    user_num, item_num = ratings.shape
    pred = np.zeros(ratings.shape)
    k_mat_1d = np.zeros(shape=user_num)

    for user in range(user_num):
        row = similarity[user]
        neighbours_idx = sorted(range(len(row)), key=lambda j: row[j], reverse=True)[0:k]
        for similar_user, sim_factor in zip(neighbours_idx, row[neighbours_idx]):
            k_mat_1d[user] += abs(sim_factor)
            for i in range(item_num):
                if ratings[user][i] > 0:
                    continue

                if ratings[similar_user][i] > 0:
                    pred[user][i] += sim_factor * ratings_diff[similar_user][i]

    return pred / np.array([k_mat_1d]).T


def predict_ksimilar_user_clust(ratings, ratings_diff, similarity, k=20):
    user_num, item_num = ratings.shape
    pred = np.zeros(ratings.shape)
    k_mat_1d = np.zeros(shape=user_num)

    for user in range(user_num):
        row = similarity[user]
        neighbours = sorted(row, key=lambda j: j['sim_factor'], reverse=True)[0:k]
        for n in neighbours:
            similar_user, sim_factor = n['user_id'], n['sim_factor']
            k_mat_1d[user] += abs(sim_factor)
            for i in range(item_num):
                if ratings[user][i] > 0:
                    continue

                if ratings[similar_user][i] > 0:
                    pred[user][i] += sim_factor * ratings_diff[similar_user][i]

    return pred / np.array([k_mat_1d]).T


def predict_ksimilar_items(ratings, ratings_diff, similarity, k=20):
    user_num, item_num = ratings.shape
    pred = np.zeros(ratings.shape)
    k_mat_1d = np.zeros(shape=item_num)

    for item in range(item_num):
        row = similarity[item]
        neighbours_idx = sorted(range(len(row)), key=lambda j: row[j], reverse=True)[0:k]
        for similar_item, sim_factor in zip(neighbours_idx, row[neighbours_idx]):
            k_mat_1d[item] += abs(sim_factor)
            for user in range(user_num):
                # if user didn't watch an item or already watched similar_item
                if ratings[user][item] <= 0 or ratings[user][similar_item] > 0:
                    continue

                pred[user][similar_item] += sim_factor * ratings_diff[user][item]

    for i in range(len(k_mat_1d)):
        if k_mat_1d[i] == 0:
            k_mat_1d[i] = 1

    return pred / k_mat_1d


def predict_clust(ratings, similarity, type='user', k_similar=20):
    pred = None
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1).reshape(-1, 1)
        ratings_diff = (ratings - mean_user_rating)

        pred = mean_user_rating + predict_ksimilar_user_clust(ratings, ratings_diff, similarity, k=k_similar)

    return pred


def predict(ratings, similarity, type='user', k_similar=None):
    pred = None
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1).reshape(-1, 1)
        ratings_diff = (ratings - mean_user_rating)

        if k_similar is None:
            pred = mean_user_rating + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        else:
            pred = mean_user_rating + predict_ksimilar_user(ratings, ratings_diff, similarity, k=k_similar)

    elif type == 'item':
        mean_item_rating = ratings.mean(axis=0).reshape(-1, 1).T
        ratings_diff = (ratings - mean_item_rating)
        if k_similar is None:
            pred = mean_item_rating + ratings_diff.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
        else:
            pred = mean_item_rating + predict_ksimilar_items(ratings, ratings_diff, similarity, k=k_similar)

    elif type == 'user-item':
        if k_similar is None:
            raise Exception('k_similar cannot be Nonen for user-item')
        mean_user_rating = ratings.mean(axis=1).reshape(-1, 1)
        ratings_diff = (ratings - mean_user_rating)
        pred_user = mean_user_rating + predict_ksimilar_user(ratings, ratings_diff, similarity['user'], k=k_similar)

        mean_item_rating = ratings.mean(axis=0).reshape(-1, 1).T
        ratings_diff = (ratings - mean_item_rating)
        pred_item = mean_item_rating + predict_ksimilar_items(ratings, ratings_diff, similarity['item'], k=k_similar)

        pred = (pred_user + pred_item) / 2

    return pred


def get_recommendations(user_item, user_prediction, n):
    recomm = []
    watched = []

    for i in range(len(user_item)):
        recomm.append([])
        watched.append([])
        for j in range(len(user_item[0])):
            if user_item[i][j] > 0:
                watched[i].append({"item_id": j, "rating": user_item[i][j]})
            else:
                recomm[i].append({"item_id": j, "pred_rating": user_prediction[i][j]})

    for i in range(len(recomm)):
        recomm[i].sort(key=lambda o: o["pred_rating"], reverse=True)
        recomm[i] = recomm[i][:n]

    for i in range(len(watched)):
        watched[i].sort(key=lambda o: o["rating"], reverse=True)

    return recomm, watched


def print_user_recommendations(items, recommendations, watched_movies, users_num=10):
    for i in range(users_num):
        print("====== User with id: {}".format(i))
        print("\tWatched:")
        for idx, watched_obj in enumerate(watched_movies[i]):
            id, rating = watched_obj['item_id'], watched_obj["rating"]
            title = items[id][1]
            print("\t{}. id: {}, rating: {}, title: {}".format(idx + 1, id, rating, title))

        print("\tRecommended:")
        for idx, recomm_obj in enumerate(recommendations[i]):
            id, rating = recomm_obj['item_id'], recomm_obj["pred_rating"]
            title = items[id][1]
            print("\t{}. id: {}, pred_rating: {}, title: {}".format(idx + 1, id, rating, title))


def pearson(a, b):
    p = pearsonr(a, b)[0]
    if p == math.nan:
        p = -1

    return p


def jaccard_sim2(a, b):
    p = b[a > 1]
    denominator = math.sqrt(len(a[a > 0]) * len(b[b > 0]))
    if denominator == 0:
        return -1
    return len(p[p > 0]) / denominator


def jaccard_sim(a, b):
    p = b[a > 1]
    common = len(p[p > 0])
    a_movies = len(a[a > 0])
    b_movies = len(b[b > 0])
    denominator = (a_movies + b_movies - common)
    if denominator == 0:
        return -1
    return common / denominator


def transform_recomm(recomm):
    return [[obj['item_id'] for obj in recomm[i]] for i in range(len(recomm))]


def ex1(plot_charts=True, verbose=True):
    """
    Simple CF using item-based and user-based method
    :param plot_charts:
    :param verbose:
    :return:
    """
    # load data
    ml100k_item_filename = 'ml-100k/u.item'
    ml100k_filename_full = 'ml-100k/u.data'
    ml100k_filename = 'ml-100k/ua.base'
    ml100k_filename_test = 'ml-100k/ua.test'

    items = DatasetManager.load_csv(ml100k_item_filename, delimeter='|', encoding='ISO-8859-1')
    data_full = DatasetManager.load_csv(ml100k_filename_full)
    data = DatasetManager.load_csv(ml100k_filename)
    test_data = DatasetManager.load_csv(ml100k_filename_test)

    # train test split
    train, test = DatasetManager.train_test_split(data_full, shuffle=False)

    # user_item = DatasetManager.transform_to_user_item_mat(data, verbose=True)
    # user_item_test = DatasetManager.transform_to_user_item_mat(test_data, verbose=True)

    user_item = DatasetManager.transform_to_user_item_mat(train, verbose=True)
    user_item_test = DatasetManager.transform_to_user_item_mat(test, verbose=True)

    # calculate similarities for user and item
    # metric = pearson
    metric = 'cosine'
    # metric = 'correlation'
    # metric = watched_movies

    user_similarity = pairwise_distances(user_item, metric=jaccard_sim)
    item_similarity = pairwise_distances(user_item.T, metric=jaccard_sim2)

    # predict
    user_prediction = predict(user_item, user_similarity, type='user', k_similar=20)
    item_prediction = predict(user_item, item_similarity, type='item', k_similar=20)
    user_item_prediction = user_prediction + item_prediction

    # get top n recommendations
    topn = 10
    recommendations_usr, watched_movies_usr = get_recommendations(user_item, user_prediction, n=topn)
    recommendations_itm, watched_movies_itm = get_recommendations(user_item, item_prediction, n=topn)
    recommendations_usr_item, watched_movies_usr_itm = get_recommendations(user_item, user_item_prediction, n=topn)

    if verbose:
        users_num = 3
        print("User based: ")
        print_user_recommendations(items, recommendations_usr, watched_movies_usr, users_num)
        print()
        print("Items based: ")
        print_user_recommendations(items, recommendations_itm, watched_movies_itm, users_num)
        print()
        print("User-item: ")
        print_user_recommendations(items, recommendations_usr_item, watched_movies_usr_itm, users_num)

    # evaluate
    evaluator = Evaluator(user_item_test)

    recomm_usr_mat = transform_recomm(recommendations_usr)
    recomm_itm_mat = transform_recomm(recommendations_itm)
    recomm_usr_itm_mat = transform_recomm(recommendations_usr_item)

    eval_user = evaluator.eval(recomm_usr_mat, user_prediction)
    eval_item = evaluator.eval(recomm_itm_mat, item_prediction)
    eval_user_item = evaluator.eval(recomm_usr_itm_mat, user_item_prediction)

    print("Eval user-based")
    print(eval_user)
    print()
    print("Eval item-based")
    print(eval_item)
    print()
    print("Eval user-item:")
    print(eval_user_item)


def pairwise_distances_clust(user_item, cluster_data, metric, sort_optim=True):
    size, _ = user_item.shape
    sim = np.zeros(shape=(size, size))
    indexes = None
    if sort_optim:
        indexes = dict()
        for key in cluster_data['groups']:
            indexes[key] = 0

    def _get_clust(user):
        return cluster_data['labels'][user]

    def _get_mates(cl_num):
        return cluster_data['groups'][cl_num]

    def _get_mates_by_usr(user):
        cl_num = _get_clust(user)
        return cluster_data['groups'][cl_num]

    if sort_optim:
        for user in range(size):
            cl = _get_clust(user)
            mates = _get_mates(cl)
            start_idx = indexes[cl]
            for j in range(start_idx, len(mates)):
                mate = mates[j]
                if user == mate:
                    indexes[cl] = j + 1
                    continue

                sim[user][mate] = metric(user_item[user], user_item[mate])

    else:
        for i in range(size):
            for j in _get_mates_by_usr(i):
                if i > j or i == j:
                    continue

                sim[i][j] = metric(user_item[i], user_item[j])

    return sim + sim.T


def group_users(cluster_labels, sort=True):
    groups = dict()

    for usr_idx, clust_num in enumerate(cluster_labels):
        if clust_num not in groups:
            groups[clust_num] = list()
        groups[clust_num].append(usr_idx)

    for key in groups.keys():
        sorted(groups[key])

    return groups


def compare_sim(user_similarity_1, user_similarity_2):
    for i in range(len(user_similarity_1)):
        for j in range(len(user_similarity_1)):
            if user_similarity_1[i][j] != 0:
                if abs(user_similarity_1[i][j] - user_similarity_2[i][j]) > 0.00000001:
                    print('{}:{} {} vs {}'.format(i, j, user_similarity_1[i][j], user_similarity_2[i][j]))


def ex2(plot_charts=True, verbose=True):
    """
    Optimized calc using clusters
    :param plot_charts:
    :param verbose:
    :return:
    """
    # load data
    ml100k_filename = 'ml-100k/u.data'
    data = DatasetManager.load_csv(ml100k_filename)

    # train test split
    train, test = DatasetManager.train_test_split(data, shuffle=False)

    # pre process
    user_item = DatasetManager.transform_to_user_item_mat(train, verbose=verbose)
    user_item_test = DatasetManager.transform_to_user_item_mat(test, verbose=verbose)

    # clusterize users
    k = 3
    cluster_labels, _ = clusterize(user_item, k, plot_charts)
    cluster_groups = group_users(cluster_labels, sort=True)
    cluster_data = {
        'labels': cluster_labels,
        'groups': cluster_groups
    }

    # calculate similarities using clusters
    time = LogTime('pairwise_distances_clust')
    user_similarity_1 = pairwise_distances_clust(user_item, cluster_data, metric=jaccard_sim, sort_optim=True)
    time.finish()

    time = LogTime('pairwise_distances')
    user_similarity_2 = pairwise_distances(user_item, metric=jaccard_sim)
    time.finish()

    compare_sim(user_similarity_1, user_similarity_2)

    # predict using clusters
    def transform_sparse_sim(sparse_sim):
        sim = list()
        for i in range(len(sparse_sim)):
            sim.append([])
            for j in range(len(sparse_sim)):
                if sparse_sim[i][j] != 0:
                    sim[i].append({
                        'user_id': j,
                        'sim_factor': sparse_sim[i][j]
                    })
        return sim

    user_similarity_1 = transform_sparse_sim(user_similarity_1)
    time = LogTime('predict_clust')
    user_prediction_clus = predict_clust(user_item, user_similarity_1, type='user', k_similar=20)
    time.finish()

    time = LogTime('predict')
    user_prediction = predict(user_item, user_similarity_2, type='user', k_similar=20)
    time.finish()

    # recomm
    recommendations_usr_clus, watched_movies_usr_clus = get_recommendations(user_item, user_prediction_clus, n=10)
    recommendations_usr, watched_movies_usr = get_recommendations(user_item, user_prediction, n=10)

    # evaluate
    evaluator = Evaluator(user_item_test)
    recomm_usr_mat_clus = transform_recomm(recommendations_usr_clus)
    recomm_usr_mat = transform_recomm(recommendations_usr)

    eval_user_clus = evaluator.eval(recomm_usr_mat_clus, user_prediction_clus)
    eval_user = evaluator.eval(recomm_usr_mat, user_prediction)

    print("Eval user-based (with clust)")
    print(eval_user_clus)
    print()
    print("Eval user-based (without clust)")
    print(eval_user)


if __name__ == '__main__':
    # ex1()
    ex2(plot_charts=False)
