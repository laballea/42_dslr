import getopt, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.common import load_data, error



def scatterplot(data: pd.DataFrame):
    data = data.dropna()
    columns_name = ["Hogwarts House", "Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes",
    "History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    sns.pairplot(data[[columns_name[0]] + columns_name[1:8]], hue="Hogwarts House")
    sns.pairplot(data[[columns_name[0]] + columns_name[8:]], hue="Hogwarts House")
    plt.show()


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:", ["file="])
    except getopt.GetoptError as inst:
        error(inst)

    try:
        for opt, arg in opts:
            if opt in ["-f", "--file"]:
                scatterplot(load_data(arg))
    except Exception as inst:
        error(inst)


if __name__ == "__main__":
    main(sys.argv[1:])