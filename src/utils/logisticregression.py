import numpy as np
from tqdm import tqdm
import random

from utils.utils_ml import intercept_, batch
from utils.metrics import cross_entropy
from utils.common import error
from utils.colors import colors


class LogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """


    def __init__(self, theta: np.ndarray, alpha: float=0.001, max_iter: int=1000, reg: str="l2", lambda_: float=1.0, gradient: str="batch", batch_size: int=32):
        self.supported_regularization = {'l2': self.ltwo, 'l1': self.lone}

        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.lambda_ = lambda_
        self.reg = reg
        self.reg_fct = self.supported_regularization["l2"]
        self.eps = 1e-15
        self.supported_gradient = {
            "batch":self.gradient,
            "mini_batch":self.mini_batch_gradient,
            "stohastic":self.stohastic_gradient,
        }
        self.batch_size = batch_size
        if gradient not in self.supported_gradient:
            error("unsupported gradient")
        else:
            self.gradient_fct = self.supported_gradient[gradient]
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

    def mini_batch_gradient(self, x: np.ndarray, y: np.ndarray):
        try:
            idx = random.randint(0, len(self.batch_x) - 1)
            return self.gradient(self.batch_x[idx], self.batch_y[idx])
        except Exception as inst:
            raise inst

    def stohastic_gradient(self, x: np.ndarray, y: np.ndarray):
        try:
            idx = random.randint(0, x.shape[0] - 1)
            x, y = np.array([x[idx]]), np.array([y[idx]])
            m, n = x.shape
            y_hat = self.predict_(x)
            x = intercept_(x)
            theta_ = self.theta.copy()
            theta_[0] = 0
            return (1 / m) * (x.T.dot(y_hat - y) + (self.lambda_ * theta_))
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

    def fit_(self, x: np.ndarray, y: np.ndarray, x_test: np.ndarray=None, y_test: np.ndarray=None, fct_metrics=cross_entropy):
        try:
            metrics_tr, metrics_cv = [], []
            self.batch_x, self.batch_y = batch(x, y, m=self.batch_size)
            for _ in tqdm(range(self.max_iter), leave=False):
                grdt = self.gradient_fct(x, y)
                self.theta = self.theta - (grdt * self.alpha)
                metrics_tr.append(fct_metrics(y, self.predict_(x)))
                if x_test is not None and y_test is not None:
                    metrics_cv.append(fct_metrics(y_test, self.predict_(x_test)))
            return metrics_tr, metrics_cv
        except Exception as inst:
            raise inst
