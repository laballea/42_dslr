import numpy as np
from utils.statistician import Statistician
import getopt, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import matplotlib.lines as mlines

def load_data(path: str):
    with open(path, "r") as stream:
        try:
            data = pd.read_csv(stream)
        except Exception as inst:
            print(inst)
            sys.exit(2)
    return data

def histogram(data: pd.DataFrame):
    data = data.dropna()
    size = math.ceil(math.sqrt(len(data.columns[6:])))
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(16,8))
    fig.tight_layout()
    for idx, col_name in enumerate(data.columns[6:]):
        usefull = data[["Hogwarts House", col_name]]
        sns.histplot(data=usefull, x=col_name, hue="Hogwarts House", kde=True, multiple="stack", stat="density", ax=axs[idx % size][math.floor(idx / size)], legend=True if idx == 0 else False)
        axs[idx % size][math.floor(idx / size)].set_xlabel(col_name)
    plt.show()

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:", ["file="])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ["-f", "--file"]:
            histogram(load_data(arg))


if __name__ == "__main__":
    main(sys.argv[1:])