import numpy as np


def perf_measure(y, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    y = np.reshape(y, (len(y)))
    y_hat = np.reshape(y_hat, (len(y_hat)))
    types = list(dict.fromkeys(y))
    for i in range(len(y_hat)):
        if y[i] == y_hat[i] == types[0]:
            TP += 1
        if y_hat[i] == types[0] and y[i] != y_hat[i]:
            FP += 1
        if y[i] == y_hat[i] == types[1]:
            TN += 1
        if y_hat[i] == types[1] and y[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tp, fp, tn, fn = perf_measure(y, y_hat)
        return (tp + tn) / (tp + fp + tn + fn)
    except Exception as inst:
        print(inst)
        return None


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tp, fp, tn, fn = perf_measure(y, y_hat)
        return (tp) / (tp + fp)
    except Exception as inst:
        print(inst)
        return None


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tp, fp, tn, fn = perf_measure(y, y_hat)
        return (tp) / (tp + fn)
    except Exception as inst:
        print(inst)
        return None


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    precision = precision_score_(y, y_hat)
    recall = recall_score_(y, y_hat)
    try:
        return (2 * precision * recall) / (precision + recall)
    except Exception as inst:
        print(inst)
        return None
