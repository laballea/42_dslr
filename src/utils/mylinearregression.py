import math
import warnings
import numpy as np
from tqdm import tqdm

class MyLinearRegressionException(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)


class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, progress_bar=False):
        if (not isinstance(alpha, float) and not isinstance(alpha, int)) or alpha <= 0:
            raise MyLinearRegressionException("MyLinearRegressionException: Alpha must be a float > 0")
        self.alpha = float(alpha)
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise MyLinearRegressionException("MyLinearRegressionException: max_iter must be an int > 0")
        self.max_iter = max_iter
        if isinstance(progress_bar, bool):
            self.progress_bar = progress_bar
        else:
            self.progress_bar = False
        if len(thetas) == 0:
            raise MyLinearRegressionException("MyLinearRegressionException: Bad thetas")
        if isinstance(thetas, np.ndarray):
            self.thetas = thetas.astype('float64')
        else:
            self.thetas = np.array(thetas,dtype='float64').reshape(-1,1)

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of dimension m * 1.
        Returns:
            y_hat as a numpy.array, a vector of dimension m * 1.
        """
        # try:
        # x_1 = np.c_[np.ones(x.shape[0]), x]
        # if x.shape[1] == self.thetas.shape[0]: # (_,n) (n, _)
        #     return x.dot(self.thetas)
        # return x_1.dot(self.thetas)
        # except Exception:
        #     return None
        if (not isinstance(x, np.ndarray) or not isinstance(self.thetas, np.ndarray)):
            print("Error in predict_: not numpy.array")
            return None
        if (x.size == 0 or self.thetas.size == 0):
            print("Error in predict_: empty array.")
            return None
        if (len(self.thetas) != x.shape[1] + 1):
            print("Error in predict_: invalid shape.")
            return None
        x = np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
        return x.dot(self.thetas)

    def loss_elem_(self, y, y_hat):
        """
        Description:  Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)\
            or y.shape[-1] == 0 or y_hat.shape[-1] == 0:
            return None
        if y.shape != y_hat.shape:
            return None
        try:
            return((y_hat - y) * (y_hat - y))
        except Exception:
            return None

    def loss_(self, y, y_hat):
        """
        Description: Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)\
            or y.shape[-1] == 0 or y_hat.shape[-1] == 0:
            return None
        if y.shape != y_hat.shape:
            return None
        try:
            return self.loss_elem_(y, y_hat).sum() / (2 * len(y))
        except Exception:
            return None

    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y and update thetas
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        Returns:
            None
        """
        # warnings.filterwarnings("error")
        if not isinstance(x,np.ndarray) or not isinstance(y, np.ndarray):
            print("Error: x or y are not good Numpy.ndarray.")
            return 
        if len(x) == 0 or len(y) == 0:
            print("Error: x or y are empty.")
            return 
        try:
            with warnings.catch_warnings():
                list_mse = []
                for _ in tqdm(range(self.max_iter), leave=False):
                    gradien = self.gradien_(x, y)
                    self.thetas = self.thetas - (self.alpha * gradien)
                    mse = MyLinearRegression.mse_(y, self.predict_(x))
                    list_mse.append(mse)
                return list_mse
        except Exception as e:
            raise MyLinearRegressionException(e)
    
    def gradien_(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
            The three arrays must have the compatible dimensions.
        Args:
            x: has to be an numpy.array, a matrix of dimension m * n.
            y: has to be an numpy.array, a vector of dimension m * 1.
        Return:
            The gradient as a numpy.array, a vector of dimensions n * 1,
                containg the result of the formula for all j.
            None if x, y, or theta are empty numpy.array.
            None if x, y and theta do not have compatible dimensions.
            None if x, y or theta is not of expected type.
        Raises:
            This function should not raise any Exception.
        """
        if (not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(self.thetas, np.ndarray)):
            print("Error in gradien_: not numpy.array")
            return None
        if (len(y) != len(x) or self.thetas.shape[0] != x.shape[1] + 1):
            print(f"Error in gradien_: len(y):{len(y)} != len(x):{len(x)} or thetas.shape[0]:{self.thetas.shape[0]} != x.shape[1]+1 :{x.shape[1]+1}")
            return None
        try:
            fct = 1 / len(x)
            x_hat = self.predict_(x)
            x = np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1).T
            return np.array(fct * (x.dot((x_hat - y))))
        except Exception as e:
            print(e)
            return None


    #****************************************************************
    # Class' Methods
    #****************************************************************

    def mse_(y, y_hat):
        """
        Description:
            Calculate the MSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
            mse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            loss_elem = (y_hat - y) * (y_hat - y)
            return loss_elem.sum() / len(y)
        except Exception:
            return None

    def rmse_(y, y_hat):
        """
        Description:
            Calculate the MSE between the predicted output and the real output.
        Args:
            y: has to be a numpy.array, a vector of dimension m * 1.
            y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
            mse: has to be a float.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exceptions.
        """
        try:
            return math.sqrt(MyLinearRegression.mse_(y, y_hat))
        except Exception:
            return None
