import getopt, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.common import load_data, error


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
        error(inst)
    try:
        for opt, arg in opts:
            if opt in ["-f", "--file"]:
                scatterplot(load_data(arg))
    except Exception as inst:
        error(inst) 


if __name__ == "__main__":
    main(sys.argv[1:])