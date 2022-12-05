import numpy as np


class Normalizer():
    def __init__(self, X=None):
        if X is not None:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
        else:
            self.mean_ = None
            self.std_ = None
        
    def norme(self, X):
        try:
            X_tr = np.copy(X)
            X_tr -= self.mean_
            X_tr /= self.std_
            return X_tr
        except Exception:
            return 0

    def inverse(self, X_tr):
        try:
            X = np.copy(X_tr)
            X *= self.std_
            X += self.mean_
            return X
        except Exception:
            return np.array([[0.0]])
