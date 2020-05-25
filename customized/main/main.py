import math

import numpy as np
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

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


def predict(ratings, similarity, type='user', k_similar=None):
    # TODO add user-item
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
    return pearsonr(a, b)[0]


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
    # TODO add custom metric
    # metric = pearson
    metric = 'cosine'
    # metric = 'correlation'
    # metric = watched_movies

    user_similarity = pairwise_distances(user_item, metric=jaccard_sim)
    item_similarity = pairwise_distances(user_item.T, metric=jaccard_sim)

    # predict
    user_prediction = predict(user_item, user_similarity, type='user', k_similar=20)
    item_prediction = predict(user_item, item_similarity, type='item', k_similar=20)

    # get top n recommendations
    topn = 10
    recommendations_usr, watched_movies_usr = get_recommendations(user_item, user_prediction, n=topn)
    recommendations_itm, watched_movies_itm = get_recommendations(user_item, item_prediction, n=topn)

    if verbose:
        users_num = 3
        print("User based: ")
        print_user_recommendations(items, recommendations_usr, watched_movies_usr, users_num)
        print()
        print("Items based: ")
        print_user_recommendations(items, recommendations_itm, watched_movies_itm, users_num)

    # evaluate
    evaluator = Evaluator(user_item_test)

    def transform_recomm(recomm):
        return [[obj['item_id'] for obj in recomm[i]] for i in range(len(recomm))]

    recomm_usr_mat = transform_recomm(recommendations_usr)
    recomm_itm_mat = transform_recomm(recommendations_itm)

    eval_user = evaluator.eval(recomm_usr_mat, user_prediction)
    eval_item = evaluator.eval(recomm_itm_mat, item_prediction)

    print("Eval user-based")
    print(eval_user)
    print()
    print("Eval item-based")
    print(eval_item)


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

    # pre process
    user_item = DatasetManager.transform_to_user_item_mat(data, verbose=verbose)

    # users_mean_ratings = np.mean(user_item, axis=1)
    # items_mean_ratings = np.mean(user_item, axis=0)
    # user_item_centered = user_item - users_mean_ratings.reshape(-1, 1)

    # clusterize users
    k = 3
    cluster_labels, _ = clusterize(user_item, k, plot_charts)

    # calculate similarities for user and item


if __name__ == '__main__':
    ex1()
    # ex2()
