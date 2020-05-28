from sklearn.metrics.pairwise import cosine_distances

from customized.main.Evaluator import Evaluator
from customized.main.models.Model import Model
from customized.main.utils.Timed import Timed
from customized.main.similarity import cosine_watched_sim, jaccard_watched_sim, cosine_classic_sim

BUILDIN_METRICS = {
    'jaccard_watched': jaccard_watched_sim,
    'cosine_watched': cosine_watched_sim,
    'cosine_ratings': cosine_classic_sim
}


class BaseCF(Model):

    def __init__(self, similarity_metric, k_similar, top_n, verbose):
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
    def recommend_all(self):
        pred = self.predict_all()
        recomm, _ = self._recommend_all(pred)

        return recomm

    def test(self, test_set):
        evaluator = Evaluator(test_set)
        pred = self.predict_all()
        recomm, _ = self._recommend_all(pred)
        recomm = self._transform_recomm(recomm)

        return evaluator.eval(recomm, pred)

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

    def _transform_recomm(self, recomm):
        return [[obj['item_id'] for obj in recomm[i]] for i in range(len(recomm))]
