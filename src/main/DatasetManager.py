import csv
import os
import random

import numpy as np


class DatasetManager:

    def __init__(self):
        pass

    @classmethod
    def load_csv(cls, filename, delimeter='\t', encoding='UTF-8', realative=True):
        """
        :param filename: path inside data folder
        :return:
        """
        if realative:
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, '../../data/{}'.format(filename))
        data = []

        with open(filename, newline='\n', encoding=encoding) as csvfile:
            reader = csv.reader(csvfile, delimiter=delimeter)

            for row in reader:
                data.append(row)

        return data

    @classmethod
    def train_test_split(cls, ratings, max_test_rat_num=10, shuffle=True, rand_seed=None, user_idx=0):
        total_rat_num = len(ratings)
        train = list()
        test = list()
        users_rat = dict()

        if shuffle:
            if rand_seed is None:
                random.shuffle(ratings)
            else:
                random.shuffle(ratings, lambda: rand_seed)

        for row in ratings:
            user_id = row[user_idx]

            if user_id not in users_rat or \
               users_rat[user_id] < max_test_rat_num:
                if user_id not in users_rat:
                    users_rat[user_id] = 0
                test.append(row)
                users_rat[user_id] += 1
            else:
                train.append(row)

        print('test size: {} %'.format(100 * len(test) / total_rat_num))
        return train, test

    @classmethod
    def transform_to_user_item_mat(cls, data, user_idx=0, item_idx=1, rating_idx=2, verbose=False):
        """
        Transform to user - item table with ratings
        :param verbose:
        :param rating_idx:
        :param item_idx:
        :param user_idx:
        :param data:
        :return: data_item matrix
        """
        data = cls.__preprocess(data)
        user_num = np.sort(data[:, user_idx])[-1] + 1
        item_num = np.sort(data[:, item_idx])[-1] + 1
        if verbose:
            print('User number: {}, item number: {}'.format(user_num, item_num))
        data_item = np.zeros(shape=(user_num, item_num))

        for row in data:
            user_id = row[user_idx]
            item_id = row[item_idx]
            rating = row[rating_idx]
            data_item[user_id][item_id] = rating

        return data_item

    @classmethod
    def __preprocess(cls, data):
        data = cls.__cast_to_int(data)
        data = np.array(data)
        data = data - np.array([1, 1, 0, 0])

        return data

    @classmethod
    def __cast_to_int(cls, data):
        """
        Cast all elements to int
        :param data:
        :return:
        """
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = int(data[i][j])

        return data
