import numpy as np


"""Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of dimension m * n,
        containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
def add_polynomial_features(x, power):
    try:
        if (not isinstance(x, np.ndarray) or (not isinstance(power, int) and not isinstance(power, list))):
            print("Invalid type")
            return None
        if (isinstance(power, list) and len(power) != x.shape[1]):
            print(f"Error in add_polynomial_features:len(power):{len(power)} != x.shape[1]:{x.shape[1]}")
            return None
        result = x.copy()
        if not isinstance(power, list):
            for po in range(2, power + 1):
                for col in x.T:
                    result = np.concatenate((result, (col**po).reshape(-1, 1)), axis=1)
        else:
            for col, power_el in zip(x.T, power):
                for po in range(2, power_el + 1):
                    result = np.concatenate((result, (col**po).reshape(-1, 1)), axis=1)
        return np.array(result)
    except Exception as inst:
        print(inst)
        return None