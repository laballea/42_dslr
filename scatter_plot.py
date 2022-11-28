
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import getopt
import sys

# used_features = ['Best Hand', 'Arithmancy', 'Astronomy', 'Herbology',
    #    'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
    #    'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
    #    'Care of Magical Creatures', 'Charms', 'Flying']

def load_data(path: str):
    """ load .csv file with path and return Dataframe of the dataset and header of the dataset droped by inused columns"""
    with open(path, "r") as stream:
        try:
            data = pd.read_csv(stream)
        except Exception as inst:
            print(inst)
            sys.exit(2)
    data.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday'], axis = 1, inplace = True)
    return data, data.columns

def display(file, features):

    data, used_features = load_data(file)
    data = data.dropna()
    if features != "All":
        if features in used_features:
            y = data[features]
            x = np.arange(len(y))
            plt.scatter(x, y, label=features)
            plt.title(features)
            plt.show()
        else:
            print(f"Error, feature {features}, not in the list : ")
            print(used_features)
    else:
        scatter_matrix(data.iloc[:,6:], alpha = 1, figsize = (len(used_features), len(used_features)), diagonal = 'hist',)
        plt.show()

def main(argv):
    file = None
    features = 'All'
    try:
        opts, _ = getopt.getopt(argv, "f:e:a:", ["file=", "feature=", "all"])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    
    for opt, arg in opts:
        print(opt,arg)
        if opt in ["-f", "--file"]:
            file = arg
        if opt in ["-e", "--feature"]:
            features = str(arg)
        if opt in ["-a", "--all"]:
            features = 'All'
    display(file, features)

if __name__ == "__main__":
    main(sys.argv[1:])