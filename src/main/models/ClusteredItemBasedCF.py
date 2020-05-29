import numpy as np
from sklearn.cluster import KMeans

from src.main.models.BaseCF import BaseCF
from src.main.utils.Timed import Timed


# TODO create common class
class ClusteredItemBasedCF(BaseCF):
    CLUST_METHODS = ['kmeans']

    def __init__(self, similarity_metric='cosine_watched', k_similar=9, top_n=10, clust_method='kmeans', n_clusters=4,
                 verbose=False):
        """

        :param similarity_metric:
        :param k_similar:
        :param top_n:
        :param clust_method:
        :param n_clusters:
        :param verbose:
        """
        super().__init__(similarity_metric, k_similar, top_n, verbose)
        if clust_method not in ClusteredItemBasedCF.CLUST_METHODS:
            raise ValueError('This clusterization method is not supported')
        self.clust_method = clust_method
        self.n_clusters = n_clusters
        self.cluster_data = None

    @Timed.timed
    def fit(self, train_set):
        self.train_set = train_set
        train_set_trans = train_set.T
        self._clusterize(train_set_trans)
        self.similarity = self._calc_sim(train_set_trans)

    @Timed.timed
    def predict_all(self):
        if self.train_set is None or self.similarity is None:
            raise Exception('You have to fit model before prediction')
        mean_item_rating = self.train_set.mean(axis=0).reshape(-1, 1).T
        ratings_diff = (self.train_set - mean_item_rating)

        return mean_item_rating + self._predict_ksimilar_users_clust(ratings_diff)

    def _predict_ksimilar_users_clust(self, ratings_diff):
        user_num, item_num = self.train_set.shape
        pred = np.zeros(self.train_set.shape)
        k_mat_1d = np.zeros(shape=item_num)

        for item in range(item_num):
            row = self.similarity[item]
            neighbours = sorted(row, key=lambda j: j['sim_factor'], reverse=True)[0:self.k_similar]
            for n in neighbours:
                similar_item, sim_factor = n['item_id'], n['sim_factor']
                k_mat_1d[item] += abs(sim_factor)
                for user in range(user_num):
                    # if user didn't watch an item or already watched similar_item
                    if self.train_set[user][item] <= 0 or self.train_set[user][similar_item] > 0:
                        continue

                    pred[user][similar_item] += sim_factor * ratings_diff[user][item]

        for i in range(len(k_mat_1d)):
            if k_mat_1d[i] == 0:
                k_mat_1d[i] = 1

        return pred / k_mat_1d

    @Timed.timed
    def _clusterize(self, train_set_trans):
        cluster_labels, model_obj = None, None
        if self.clust_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters)
            kmeans.fit(train_set_trans)

            cluster_labels, model_obj = kmeans.labels_, kmeans

        cluster_groups = self._group_items(cluster_labels)
        self.cluster_data = {
            'labels': cluster_labels,
            'groups': cluster_groups
        }

    @staticmethod
    def _group_items(cluster_labels):
        groups = dict()

        for usr_idx, clust_num in enumerate(cluster_labels):
            if clust_num not in groups:
                groups[clust_num] = list()
            groups[clust_num].append(usr_idx)

        for key in groups.keys():
            sorted(groups[key])

        return groups

    @Timed.timed
    def _calc_sim(self, train_set_trans):
        return self._transform_sparse_sim(self._pairwise_distances_clust(train_set_trans))

    def _pairwise_distances_clust(self, train_set_trans):
        item_num, _ = train_set_trans.shape
        sim = np.zeros(shape=(item_num, item_num))
        indexes = dict()
        for key in self.cluster_data['groups']:
            indexes[key] = 0

        def _get_clust(item):
            return self.cluster_data['labels'][item]

        def _get_mates(cl_num):
            return self.cluster_data['groups'][cl_num]

        for item in range(item_num):
            cl = _get_clust(item)
            item_mates = _get_mates(cl)
            start_idx = indexes[cl]
            for j in range(start_idx, len(item_mates)):
                item_mate = item_mates[j]
                if item == item_mate:
                    indexes[cl] = j + 1
                    continue

                sim[item][item_mate] = self.similarity_metric(train_set_trans[item], train_set_trans[item_mate])

        return sim + sim.T

    @staticmethod
    def _transform_sparse_sim(sparse_sim):
        sim = list()
        size = len(sparse_sim)
        for i in range(size):
            sim.append([])
            for j in range(size):
                if sparse_sim[i][j] != 0:
                    sim[i].append({
                        'item_id': j,
                        'sim_factor': sparse_sim[i][j]
                    })
        return sim
