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

def scatterplot(data: pd.DataFrame):
    data = data.dropna()
    columns_name = ["Hogwarts House","Astronomy","Defense Against the Dark Arts"]
    data = data[columns_name]
    sns.scatterplot(data, x="Astronomy", y="Defense Against the Dark Arts", hue="Hogwarts House")
    plt.show()

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:", ["file="])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ["-f", "--file"]:
            scatterplot(load_data(arg))


if __name__ == "__main__":
    main(sys.argv[1:])