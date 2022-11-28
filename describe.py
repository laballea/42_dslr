import numpy as np
from utils.statistician import Statistician
import getopt, sys
import pandas as pd


def load_data(path: str):
    with open(path, "r") as stream:
        try:
            data = pd.read_csv(stream)
        except Exception as inst:
            print(inst)
            sys.exit(2)
    return data


def get_header(data):
    head = [str(idx) for idx in range(len(data.columns) - 1)]
    return head

def get_description(data: pd.DataFrame):
    res = {}
    for idx, col_name in enumerate(data.columns[1:]):
        value = np.array(data[col_name])
        quartile = Statistician().quartile(value)
        print(quartile)
        res[idx] = [
            Statistician().count(value),
            Statistician().mean(value),
            Statistician().std(value),
            Statistician().min(value),
            quartile[0] if quartile is not None else None,
            Statistician().median(value),
            quartile[1] if quartile is not None else None,
            Statistician().max(value),
            Statistician().var(value),
        ]
    return res

def describe(data: pd.DataFrame):
    data = data.dropna()
    data = get_description(data)
    df = pd.DataFrame(data, index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Var"])
    print(df[4:])

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:", ["file="])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ["-f", "--file"]:
            describe(load_data(arg))


if __name__ == "__main__":
    main(sys.argv[1:])