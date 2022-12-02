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
    size = math.ceil(math.sqrt(len(data.columns[6:])))
    columns_name = ["Hogwarts House","Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes",
    "History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    sns.pairplot(data[[columns_name[0]] + columns_name[1:7]], hue="Hogwarts House")
    sns.pairplot(data[[columns_name[0]] + columns_name[8:14]], hue="Hogwarts House")
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