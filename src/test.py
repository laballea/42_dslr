import numpy as np
from utils.statistician import Statistician
import getopt, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
from utils.logisticregression import LogisticRegression as LR
from utils.normalizer import Normalizer
from utils.metrics import f1_score_

def load_data(path: str):
    with open(path, "r") as stream:
        try:
            data = pd.read_csv(stream)
        except Exception as inst:
            print(inst)
            sys.exit(2)
    return data

def format_result(arr):
    result = []
    for index, row in arr.iterrows():
        result.append(row.idxmax())
    result = np.array(result).reshape(-1, 1)
    return result

def format(arr, zipcode):
    copy = arr.copy()
    copy[:, 0][copy[:, 0] != zipcode] = -1
    copy[:, 0][copy[:, 0] == zipcode] = 1
    copy[:, 0][copy[:, 0] == -1] = 0
    return copy

def one_vs_all(X, Y):
    result = pd.DataFrame()
    for label in range(len(np.unique(Y))):
        binary_y = format(Y, label)
        theta = np.zeros((X.shape[1] + 1, )).reshape(-1, 1)
        my_lr = LR(theta, alpha=0.3, max_iter=2000)
        my_lr.fit_(X, binary_y)
        y_hat = my_lr.predict_(X)
        # model["theta"][str(lambda_)][zipcode] = [float(tta) for tta in my_lr.theta]
        result[label] = y_hat.reshape(len(y_hat))
    return format_result(result), f1_score_(Y, format_result(result))

def normalize(data: pd.DataFrame):
    X = data[["Arithmancy","Astronomy", "Best Hand","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes",
    "History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]]
    Y = data[["Hogwarts House"]]
    norm_X = Normalizer(X.to_numpy(), norm="minmax").normalize().astype(float)
    norm_Y = Normalizer(Y.to_numpy(), norm="minmax").normalize().astype(float)
    return norm_X, norm_Y

def train(data: pd.DataFrame):
    data = data.dropna()
    norm_X, norm_Y = normalize(data)
    print(one_vs_all(norm_X, norm_Y))

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:", ["file="])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ["-f", "--file"]:
            train(load_data(arg))


if __name__ == "__main__":
    main(sys.argv[1:])