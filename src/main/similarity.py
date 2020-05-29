import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_watched_sim(a, b):
    p = b[a > 1]
    denominator = math.sqrt(len(a[a > 0]) * len(b[b > 0]))
    if denominator == 0:
        return -1
    return len(p[p > 0]) / denominator


def jaccard_watched_sim(a, b):
    p = b[a > 1]
    common = len(p[p > 0])
    a_movies = len(a[a > 0])
    b_movies = len(b[b > 0])
    denominator = (a_movies + b_movies - common)
    if denominator == 0:
        return -1
    return common / denominator


def cosine_classic_sim(a, b):
    return cosine_similarity(np.array([a]), np.array([b]))[0]
