import inspect

from customized.main.utils.LogTime import LogTime


class Model:

    def fit(self, train_set):
        """
        Fit model
        :param train_set: set with train data
        :return:
        """
        pass

    def recommend(self, user):
        """
        Get recommendations for a user
        :param user: user id
        :return: array with recommendations
        """
        pass

    def recommend_all(self):
        """
        Get recommendations for all users
        :return:
        """
        pass

    def predict(self, user):
        """
        Get array with predicted ratings for specified user
        :param user: user id
        :return: array with predicted ratings
        """
        pass

    def predict_all(self):
        """
        Get all predictions
        :return: 2D array with all predicted ratings
        """
        pass

    def test(self, test_set):
        """
        Calculate evaluation metrics
        :param test_set: test dataset
        :return: calculated metrics
        """
        pass
