import numpy as np
import getopt, sys
import pandas as pd
from utils.logisticregression import LogisticRegression as LR
from utils.normalizer import Normalizer
import yaml
from utils.utils_ml import add_polynomial_features
import csv

def load_data(path: str):
    with open(path, "r") as stream:
        try:
            data = pd.read_csv(stream)
        except Exception as inst:
            print(inst)
            sys.exit(2)
    return data

def format_all(arr: np.ndarray):
    """
    get an array of dimension (M, number of label) in argument representing probability of each label
    return an array of dimension (M, 1), where the best probability is choosen
    """
    result = []
    for index, row in arr.iterrows():
        result.append(row.idxmax())
    result = np.array(result).reshape(-1, 1)
    return result

def one_vs_all_predict(X, model: dict, nb_of_label: int):
    y_hat_all = pd.DataFrame()
    X_poly = add_polynomial_features(X, model["power_x"])
    for label in range(nb_of_label):
        theta = np.array(model["theta"][label]).reshape(-1, 1)

        my_lr = LR(theta, lambda_=float(model["lambda"]))

        y_hat_one = my_lr.predict_(X_poly)
        y_hat_all[label] = y_hat_one.reshape(len(y_hat_one))

    return format_all(y_hat_all)

def normalize(k_folds: tuple):
    """
    get k_folds in argument (x_train, y_train, x_test, y_test)
    and return x_train, y_train, x_test, y_test butnormalized and labelized if needed
    """
    x_train, y_train, x_test, y_test = k_folds
    x_train, y_train = Normalizer(x_train, norm="minmax").normalize().astype(float), Normalizer(y_train, norm="minmax").normalize().astype(float)
    x_test, y_test = Normalizer(x_test, norm="minmax").normalize().astype(float), Normalizer(y_test, norm="minmax").normalize().astype(float)
    return x_train, y_train, x_test, y_test


def predict(yml_file: dict, dataTest: pd.DataFrame, dataTrain: pd.DataFrame):
    model = yml_file["models"][yml_file["data"]["best_model"]]
    X_train, Y_train = parse(dataTrain)
    nb_of_label = len(np.unique(Y_train))
    normizer_X = Normalizer(X_train.to_numpy(), norm="minmax")
    normizer_Y = Normalizer(Y_train.to_numpy(), norm="minmax")

    X_test_raw = dataTest[["Astronomy", "Best Hand","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes",
        "History of Magic","Transfiguration","Potions","Charms","Flying"]]
    X_test_raw = X_test_raw.dropna()

    X_test_norm = normizer_X.normalize(X_test_raw.to_numpy()).astype(float)
    y_hat = normizer_Y.denormalize(one_vs_all_predict(X_test_norm, model, nb_of_label=nb_of_label))
    result = {
        "Index": range(len(y_hat)),
        "Hogwarts House":y_hat.reshape(len(y_hat),)
    }
    pd.DataFrame(data=result).to_csv("house.csv", index=False)

def parse(data: pd.DataFrame):
    data = data.dropna()  # remove nan from data
    X = data[["Astronomy", "Best Hand","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes",
        "History of Magic","Transfiguration","Potions","Charms","Flying"]]
    Y = data[["Hogwarts House"]]
    return X, Y

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:", ["file=", "predict", "display"])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    data_test = None
    with open("models.yml", "r") as stream:
        try:
            yml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    data_train = load_data(yml_file["data"]["data_train_path"])
    for opt, arg in opts:
        if opt in ["-f", "--file"]:
            data_test = load_data(arg)

    if data_test is None:
        raise ValueError("No data provided.")
    for opt, arg in opts:
        if opt in ["--predict"]:
            predict(yml_file, data_test, data_train)
if __name__ == "__main__":
    main(sys.argv[1:])