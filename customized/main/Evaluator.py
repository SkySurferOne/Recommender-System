import math


class Evaluator:

    def __init__(self, test):
        self.test = test
        self.test_rows = len(test)
        self.test_cols = len(test[0])
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def __rmse(self, prediction_mat):
        count = 0
        sum_res = 0.0
        for i in range(self.test_rows):
            for j in range(self.test_cols):
                if self.test[i][j] > 0:
                    sum_res += (prediction_mat[i][j] - self.test[i][j]) ** 2
                    count += 1

        return math.sqrt(sum_res / count)

    def __mae(self, prediction_mat):
        count = 0
        sum_res = 0.0
        for i in range(self.test_rows):
            for j in range(self.test_cols):
                if self.test[i][j] > 0:
                    sum_res += abs(prediction_mat[i][j] - self.test[i][j])
                    count += 1

        return sum_res / count

    def __pre_calc(self, recomm_items):
        self.TP = 0
        self.FP = 0
        self.FN = 0

        for i in range(self.test_rows):
            for j in range(len(recomm_items[i])):
                if recomm_items[i][j] < self.test_cols and \
                   self.test[i][recomm_items[i][j]] > 0:
                    self.TP += 1
                else:
                    self.FP += 1

        for i in range(self.test_rows):
            for j in range(self.test_cols):
                if self.test[i][j] > 0 and j not in recomm_items[i]:
                    self.FN += 1

    def __prec(self):
        return float(self.TP) / float(self.TP + self.FP)

    def __recall(self):
        return float(self.TP) / float(self.TP + self.FN)

    def __f1(self, prec, recall):
        if prec + recall == 0:
            f1 = 0
        else:
            f1 = 2 * prec * recall / (prec + recall)

        return f1

    def eval(self, recomm_items, prediction_mat, ret_obj=True):
        """
        Calculates precision, recall, f1 score, RMSE and MAE.
        :param recomm_items: 2D array with recommended items (ids) for users
        :param prediction_mat: Matrix with all ratings predictions
        :param ret_obj: return object if True, if not return list
        [
            user0 -> [item_id0, item_id1, ...]
            ...
        ]
        :return:
        """
        self.__pre_calc(recomm_items)
        prec = self.__prec()
        recall = self.__recall()
        f1 = self.__f1(prec, recall)
        rmse = self.__rmse(prediction_mat)
        mae = self.__mae(prediction_mat)

        if ret_obj:
            return {
                "prec": prec,
                "recall": recall,
                "f1": f1,
                "rmse": rmse,
                "mae": mae
            }
        else:
            return [prec, recall, f1, rmse, mae]
