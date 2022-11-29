import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from utils.common import is_numeric


    #Y_n = LabelEncoder().fit_transform(Y).reshape(-1, 1)

class Normalizer():
    def __init__(self, X: np.ndarray=None, norm="minmax"):
        self.supported_norm = {
            "minmax":{
                "norm":self.minmax,
                "unnorm":self.unminmax
            },
        }
        if norm not in self.supported_norm:
            raise ValueError(f"{norm} not supported")
        if X is None:
            raise ValueError(f"{norm} not supported")
        self.X = X
        self.norm_fct = self.supported_norm[norm]["norm"]
        self.unnorm_fct = self.supported_norm[norm]["unnorm"]


    def normalize(self):
        try:
            result = np.array([]).reshape(self.X.shape[0], 0)
            for col in self.X.T:
                if is_numeric(list(col)):
                    result = np.append(result, self.norm_fct(col).reshape(-1, 1), axis=1)
                else:
                    result = np.append(result, LabelEncoder().fit_transform(col).reshape(-1, 1), axis=1)
            return result
        except Exception as inst:
            raise inst

    def minmax(self, X: np.ndarray):
        try:
            result = X.copy()
            return np.array((result - min(X)) / (max(X) - min(X)))
        except Exception as inst:
            raise inst

    def unminmax(self, X: np.ndarray):
        """
        normalize matrix with minmax method
        """
        try:
            result = []
            for row_x, row_base in zip(X.T, self.X.T):
                min_r = min(row_base)
                max_r = max(row_base)
                result.append([el * (max_r - min_r) + min_r for el in row_x])
            return np.array(result).T
        except Exception as inst:
            print(inst)
            sys.exit()