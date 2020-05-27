import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from customized.main.models.BaseCF import BaseCF
from customized.main.models.Timed import Timed


class ItemBasedCF(BaseCF):

    def __init__(self, similarity_metric='cosine_watched', k_similar=20, top_n=10, verbose=False):
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
        self.similarity = pairwise_distances(train_set.T, metric=self.similarity_metric)

    @Timed.timed
    def recommend(self, user):
        #TODO
        pass

    @Timed.timed
    def predict(self, user):
        #TODO
        pass

    @Timed.timed
    def predict_all(self):
        if self.train_set is None or self.similarity is None:
            raise Exception('You have to fit model before prediction')
        mean_item_rating = self.train_set.mean(axis=0).reshape(-1, 1).T
        ratings_diff = (self.train_set - mean_item_rating)

        if self.k_similar is None:
            return mean_item_rating + ratings_diff.dot(self.similarity) / np.array(
                [np.abs(self.similarity).sum(axis=1)])
        else:
            return mean_item_rating + self._predict_ksimilar_items(ratings_diff)

    def _predict_ksimilar_items(self, ratings_diff):
        user_num, item_num = self.train_set.shape
        pred = np.zeros(self.train_set.shape)
        k_mat_1d = np.zeros(shape=item_num)

        for item in range(item_num):
            row = self.similarity[item]
            neighbours_idx = sorted(range(len(row)), key=lambda j: row[j], reverse=True)[0:self.k_similar]
            for similar_item, sim_factor in zip(neighbours_idx, row[neighbours_idx]):
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
