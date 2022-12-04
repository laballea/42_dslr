import getopt, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from utils.common import load_data, error


def histogram(data: pd.DataFrame):
    data = data.dropna()
    size = math.ceil(math.sqrt(len(data.columns[6:])))
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(16,8))
    fig.tight_layout()
    for idx, col_name in enumerate(data.columns[6:]):
        usefull = data[["Hogwarts House", col_name]]
        sns.histplot(data=usefull, x=col_name, hue="Hogwarts House", kde=True, multiple="stack", stat="density", ax=axs[idx % size][math.floor(idx / size)], legend=True if idx == 0 else False)
    plt.show()


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:", ["file="])
    except getopt.GetoptError as inst:
        error(inst)
    try:
        for opt, arg in opts:
            if opt in ["-f", "--file"]:
                histogram(load_data(arg))
    except Exception as inst:
        error(inst)


if __name__ == "__main__":
    main(sys.argv[1:])