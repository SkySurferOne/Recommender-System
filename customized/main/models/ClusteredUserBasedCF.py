from customized.main.models.BaseCF import BaseCF


class ClusteredUserBasedCF(BaseCF):

    def __init__(self, similarity_metric, k_similar, top_n, verbose):
        super().__init__(similarity_metric, k_similar, top_n, verbose)