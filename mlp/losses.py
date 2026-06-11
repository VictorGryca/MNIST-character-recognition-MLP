import numpy as np


class Losses:
    def cross_entropy(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred))
