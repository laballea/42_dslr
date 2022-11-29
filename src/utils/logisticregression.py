import numpy as np
from tqdm import tqdm
from utils.utils_ml import intercept_
import sys

class LogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """


    def __init__(self, theta, alpha=0.001, max_iter=1000, reg="l2", lambda_=1.0):
        self.supported_regularization = {'l2': self.ltwo, 'l1': self.lone}

        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.lambda_ = lambda_
        self.reg = reg
        self.reg_fct = self.supported_regularization["l2"]
        self.eps = 1e-15
        if self.reg in self.supported_regularization:
            self.reg_fct = self.supported_regularization[self.reg]
        else:
            self.lambda_ = 0.0

    def ltwo(self):
        return self.lambda_ * sum(self.theta**2)
    
    def lone(self):
        return self.lambda_ * sum(abs(self.theta))

    def sigmoid_(self, x: np.ndarray):
        try:
            return np.array(1 / (1 + np.exp(-x))).astype(float)
        except Exception as inst:
            raise inst

    def predict_(self, x: np.ndarray):
        try:
            x = intercept_(x)
            return self.sigmoid_(x.dot(self.theta))
        except Exception as inst:
            raise inst

    def loss_elem_(self, y, y_hat):
        return None

    def loss_(self, y: np.ndarray, y_hat: np.ndarray):
        try:
            eps = 1e-15
            m, n = y.shape
            ones = np.ones(y.shape)
            return float(-(1 / m) * (y.T.dot(np.log(y_hat + eps)) + (ones - y).T.dot(np.log(ones - y_hat + eps))))
        except Exception as inst:
            raise inst

    def gradient(self, x: np.ndarray, y: np.ndarray):
        try:
            m, n = x.shape
            y_hat = self.predict_(x)
            x = intercept_(x)
            theta_ = self.theta.copy()
            theta_[0] = 0
            return (1 / m) * (x.T.dot(y_hat - y) + (self.lambda_ * theta_))
        except Exception as inst:
            raise inst

    def fit_(self, x: np.ndarray, y: np.ndarray):
        try:
            historic = []
            for _ in tqdm(range(self.max_iter), leave=False):
                grdt = self.gradient(x, y)
                self.theta = self.theta - (grdt * self.alpha)
            return historic
        except Exception as inst:
            raise inst
