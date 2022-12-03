import numpy as np
import pandas as pd
import sys
from .colors import colors
import yaml

_NUMERIC_KINDS = set('buifc')


def error(msg: str="", exit: int=2, color: str=colors.red):
    print(f"{color}{msg}")
    sys.exit(exit)


def load_data(path: str):
    try:
        with open(path, "r") as stream:
            data = pd.read_csv(stream)
    except Exception as inst:
        error(inst)
    return data


def load_yml_file(path: str):
    try:
         with open(path, "r") as stream:
            data = yaml.safe_load(stream)
    except Exception as inst:
        error(inst)
    return data


def is_numeric(array: np.ndarray):
    """Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.

    Booleans, unsigned integers, signed integers, floats and complex
    numbers are the kinds of numeric datatype.

    Parameters
    ----------
    array : array-like
        The array to check.

    Returns
    -------
    is_numeric : `bool`
        True if the array has a numeric datatype, False if not.

    """
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS