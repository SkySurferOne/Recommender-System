import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from customized.main.Evaluator import Evaluator
from customized.main.models.Model import Model
from customized.main.models.Timed import Timed
from customized.main.similarity import jaccard_watched_sim, cosine_watched_sim

BUILDIN_METRICS = {
    'jaccard_watched': jaccard_watched_sim,
    'cosine_watched': cosine_watched_sim,
    'cosine_ratings': 'cosine'
}


class UserBasedCF(Model):

    def __init__(self, similarity_metric='jaccard_watched', k_similar=20, top_n=10, verbose=False):
        """

        :param similarity_metric:
        :param k_similar:
        :param top_n:
        :param verbose:
        """
        if not callable(similarity_metric) and \
                similarity_metric not in BUILDIN_METRICS:
            raise ValueError('Incorrect similarity_metric')
        if callable(similarity_metric):
            self.similarity_metric = similarity_metric
        else:
            self.similarity_metric = BUILDIN_METRICS[similarity_metric]
        self.similarity = None
        self.train_set = None
        self.verbose = verbose
        self.k_similar = k_similar
        self.top_n = top_n

    @Timed.timed
    def fit(self, train_set):
        self.train_set = train_set
        self.similarity = pairwise_distances(train_set, metric=self.similarity_metric)

    @Timed.timed
    def recommend(self, user):
        # TODO
        pass

    @Timed.timed
    def recommend_all(self):
        pred = self.predict_all()
        recomm, _ = self._recommend_all(pred)

        return recomm

    def _recommend_all(self, pred):
        user_num, item_num = self.train_set.shape
        recomm = []
        watched = []

        for i in range(user_num):
            recomm.append([])
            watched.append([])
            for j in range(item_num):
                if self.train_set[i][j] > 0:
                    watched[i].append({"item_id": j, "rating": self.train_set[i][j]})
                else:
                    recomm[i].append({"item_id": j, "pred_rating": pred[i][j]})

        for i in range(len(recomm)):
            recomm[i].sort(key=lambda o: o["pred_rating"], reverse=True)
            recomm[i] = recomm[i][:self.top_n]

        for i in range(len(watched)):
            watched[i].sort(key=lambda o: o["rating"], reverse=True)

        return recomm, watched

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
            return mean_user_rating + self.similarity.dot(ratings_diff) / np.array([np.abs(self.similarity).sum(axis=1)]).T
        else:
            return mean_user_rating + self._predict_ksimilar_users(ratings_diff)

    def test(self, test_set):
        evaluator = Evaluator(test_set)
        pred = self.predict_all()
        recomm, _ = self._recommend_all(pred)
        recomm = self._transform_recomm(recomm)

        return evaluator.eval(recomm, pred)

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

    def _transform_recomm(self, recomm):
        return [[obj['item_id'] for obj in recomm[i]] for i in range(len(recomm))]
