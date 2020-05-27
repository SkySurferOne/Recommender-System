import numpy as np
from sklearn.cluster import KMeans

from customized.main.models.BaseCF import BaseCF
from customized.main.models.Timed import Timed


class ClusteredUserBasedCF(BaseCF):
    CLUST_METHODS = ['kmeans']

    def __init__(self, similarity_metric='jaccard_watched', k_similar=20, top_n=10, clust_method='kmeans', n_clusters=3,
                 verbose=False):
        """

        :param similarity_metric:
        :param k_similar:
        :param top_n:
        :param clust_method:
        :param k_clust:
        :param verbose:
        """
        super().__init__(similarity_metric, k_similar, top_n, verbose)
        if clust_method not in ClusteredUserBasedCF.CLUST_METHODS:
            raise ValueError('This clusterization method is not supported')
        self.clust_method = clust_method
        self.n_clusters = n_clusters
        self.cluster_data = None

    @Timed.timed
    def fit(self, train_set):
        self.train_set = train_set
        self._clusterize(train_set)
        self.similarity = self._calc_sim()

    @Timed.timed
    def recommend(self, user):
        # TODO
        pass

    @Timed.timed
    def predict(self, user):
        # TODO
        pass

    @Timed.timed
    def predict_all(self):
        mean_user_rating = self.train_set.mean(axis=1).reshape(-1, 1)
        ratings_diff = (self.train_set - mean_user_rating)

        return mean_user_rating + self._predict_ksimilar_users_clust(ratings_diff)

    def _predict_ksimilar_users_clust(self, ratings_diff):
        user_num, item_num = self.train_set.shape
        pred = np.zeros(self.train_set.shape)
        k_mat_1d = np.zeros(shape=user_num)

        for user in range(user_num):
            row = self.similarity[user]
            neighbours = sorted(row, key=lambda j: j['sim_factor'], reverse=True)[0:self.k_similar]
            for n in neighbours:
                similar_user, sim_factor = n['user_id'], n['sim_factor']
                k_mat_1d[user] += abs(sim_factor)
                for i in range(item_num):
                    if self.train_set[user][i] > 0:
                        continue

                    if self.train_set[similar_user][i] > 0:
                        pred[user][i] += sim_factor * ratings_diff[similar_user][i]

        return pred / np.array([k_mat_1d]).T

    @Timed.timed
    def _clusterize(self, train_set):
        cluster_labels, model_obj = None, None
        if self.clust_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters)
            kmeans.fit(train_set)

            cluster_labels, model_obj = kmeans.labels_, kmeans

        cluster_groups = self._group_users(cluster_labels)
        self.cluster_data = {
            'labels': cluster_labels,
            'groups': cluster_groups
        }

    @staticmethod
    def _group_users(cluster_labels):
        groups = dict()

        for usr_idx, clust_num in enumerate(cluster_labels):
            if clust_num not in groups:
                groups[clust_num] = list()
            groups[clust_num].append(usr_idx)

        for key in groups.keys():
            sorted(groups[key])

        return groups

    @Timed.timed
    def _calc_sim(self):
        return self._transform_sparse_sim(self._pairwise_distances_clust())

    def _pairwise_distances_clust(self):
        size, _ = self.train_set.shape
        sim = np.zeros(shape=(size, size))
        indexes = dict()
        for key in self.cluster_data['groups']:
            indexes[key] = 0

        def _get_clust(user):
            return self.cluster_data['labels'][user]

        def _get_mates(cl_num):
            return self.cluster_data['groups'][cl_num]

        for user in range(size):
            cl = _get_clust(user)
            mates = _get_mates(cl)
            start_idx = indexes[cl]
            for j in range(start_idx, len(mates)):
                mate = mates[j]
                if user == mate:
                    indexes[cl] = j + 1
                    continue

                sim[user][mate] = self.similarity_metric(self.train_set[user], self.train_set[mate])

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
                        'user_id': j,
                        'sim_factor': sparse_sim[i][j]
                    })
        return sim
