import numpy as np


def merge_labels_2d(X, labels):
    return np.c_[X[:, 0], X[:, 1], labels]