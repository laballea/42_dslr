import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from utils.common import is_numeric



class Normalizer():
    def __init__(self, X: np.ndarray=None, norm="minmax"):
        self.supported_norm = {
            "minmax":{
                "norm":self.minmax,
                "unnorm":self.unminmax
            },
            "zscore":{
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
        self.info_col = self.get_info()

    def get_info(self):
        try:
            result = []
            for col in self.X.T:
                if is_numeric(list(col)):
                    result.append("n")
                else:
                    result.append("l")
            return result
        except Exception as inst:
            raise inst

    def labelize(self):
        try:
            result = np.array([]).reshape(self.X.shape[0], 0)
            for col, col_info in zip(self.X.T, self.info_col):
                if col_info == "n":
                    result = np.append(result, col.reshape(-1, 1), axis=1)
                elif col_info == "l":
                    result = np.append(result, LabelEncoder().fit_transform(col).reshape(-1, 1), axis=1)
            return result
        except Exception as inst:
            raise inst

    def denormalize(self, to_norm: np.ndarray=None):
        try:
            if to_norm is None:
                to_norm = self.X
            result = np.array([]).reshape(to_norm.shape[0], 0)
            for col_to_norm, col_ref, col_info in zip(to_norm.T, self.X.T, self.info_col):
                if col_info == "n":
                    result = np.append(result, self.unnorm_fct(col_to_norm, col_ref).reshape(-1, 1), axis=1)
                elif col_info == "l":
                    le = LabelEncoder()
                    le.fit_transform(col_ref)
                    le.inverse_transform(col_to_norm)
                    result = np.append(result, le.inverse_transform(col_to_norm).reshape(-1, 1), axis=1)
            return result
        except Exception as inst:
            raise inst

    def normalize(self, to_norm: np.ndarray=None):
        try:
            if to_norm is None:
                to_norm = self.X
            result = np.array([]).reshape(to_norm.shape[0], 0)
            for col_to_norm, col_ref, col_info in zip(to_norm.T, self.X.T, self.info_col):
                if col_info == "n":
                    result = np.append(result, self.norm_fct(col_to_norm, col_ref).reshape(-1, 1), axis=1)
                elif col_info == "l":
                    result = np.append(result, LabelEncoder().fit_transform(col_to_norm).reshape(-1, 1), axis=1)
            return result
        except Exception as inst:
            raise inst

    def minmax(self, to_norm: np.ndarray=None, ref: np.ndarray=None):
        try:
            result = to_norm.copy()
            return np.array((result - min(ref)) / (max(ref) - min(ref)))
        except Exception as inst:
            raise inst

    def unminmax(self, to_norm: np.ndarray=None, ref: np.ndarray=None):
        try:
            result = to_norm.copy()
            return np.array((result * (max(ref) - min(ref)) + min(ref)))
        except Exception as inst:
            raise inst

    def zscore(self, to_norm: np.ndarray=None, ref: np.ndarray=None):
        try:
            result = np.copy(to_norm)
            result -= np.mean(ref)
            result /= np.std(ref)
            return result
        except Exception as inst:
            raise inst

    def unzscore(self, to_norm: np.ndarray=None, ref: np.ndarray=None):
        try:
            result = np.copy(to_norm)
            result *= np.std(ref)
            result += np.mean(ref)
            return result
        except Exception as inst:
            raise inst

    # def unminmax(self, X: np.ndarray):
    #     """
    #     normalize matrix with minmax method
    #     """
    #     try:
    #         result = []
    #         for row_x, row_base in zip(X.T, self.X.T):
    #             min_r = min(row_base)
    #             max_r = max(row_base)
    #             result.append([el * (max_r - min_r) + min_r for el in row_x])
    #         return np.array(result).T
    #     except Exception as inst:
    #         print(inst)
    #         sys.exit()