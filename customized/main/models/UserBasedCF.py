import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from customized.main.models.BaseCF import BaseCF
from customized.main.utils.Timed import Timed


class UserBasedCF(BaseCF):

    def __init__(self, similarity_metric='jaccard_watched', k_similar=20, top_n=10, verbose=False):
        """

        :param similarity_metric:
        :param k_similar:
        :param top_n:
        :param verbose:
        """
        super().__init__(similarity_metric, k_similar, top_n, verbose)

    @Timed.timed
    def fit(self, train_set):
        self.train_set = train_set
        self.similarity = pairwise_distances(train_set, metric=self.similarity_metric)

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
        if self.train_set is None or self.similarity is None:
            raise Exception('You have to fit model before prediction')
        mean_user_rating = self.train_set.mean(axis=1).reshape(-1, 1)
        ratings_diff = (self.train_set - mean_user_rating)

        if self.k_similar is None:
            return mean_user_rating + self.similarity.dot(ratings_diff) / np.array(
                [np.abs(self.similarity).sum(axis=1)]).T
        else:
            return mean_user_rating + self._predict_ksimilar_users(ratings_diff)

    def _predict_ksimilar_users(self, ratings_diff):
        user_num, item_num = self.train_set.shape
        pred = np.zeros(self.train_set.shape)
        k_mat_1d = np.zeros(shape=user_num)

        for user in range(user_num):
            row = self.similarity[user]
            neighbours_idx = sorted(range(len(row)), key=lambda j: row[j], reverse=True)[0:self.k_similar]
            for similar_user, sim_factor in zip(neighbours_idx, row[neighbours_idx]):
                k_mat_1d[user] += abs(sim_factor)
                for i in range(item_num):
                    if self.train_set[user][i] > 0:
                        continue

                    if self.train_set[similar_user][i] > 0:
                        pred[user][i] += sim_factor * ratings_diff[similar_user][i]

        return pred / np.array([k_mat_1d]).T
