import numpy as np
import getopt, sys
import pandas as pd
import seaborn as sns
import itertools
import yaml
from tqdm import tqdm
import math
from matplotlib import pyplot as plt
import time

from utils.logisticregression import LogisticRegression as LR
from utils.normalizer import Normalizer
from utils.metrics import accuracy_score_
from utils.cleaning import cleaner
from utils.utils_ml import cross_validation, add_polynomial_features, data_spliter
from utils.common import load_data, load_yml_file, error, colors


def get_combs(rangePower: list, nb_param=1, full_comb=False):
    """
        return possible combination depending on the power range and if it is fully combinated
        when it's not, all parameter will have same power, possibility = rangePower**1
        when it is, all parameter can have either the same or not power, it's demultiply possibility, possibility = rangePower**nb_param
    """
    if not full_comb:
        return [np.full((nb_param, 1), po) for po in rangePower]
    else:
        return np.array(list(itertools.product(list(itertools.product(rangePower)), repeat=nb_param)))


def reset(powerRange: list, lambdaRange: list, X:np.ndarray, Y:np.ndarray, dataPath:str, gradient: str):
    """
    reset the yml file, all data is lost
    get in parameter powerRange, the range of power to handle
    and lambdaRange, same but lambda
    """
    models = {}
    combs = get_combs(powerRange, nb_param=X.shape[1], full_comb=False)
    number_of_label = len(np.unique(Y))
    models["data"] = {
        "best_model":None,
        "number_of_models":len(combs) * len(lambdaRange),
        "data_train_path":dataPath,
    }
    models["models"] = {}
    for comb in combs:
        for lambda_ in lambdaRange:
            comb = comb.reshape(len(comb),)
            model_name = '_'.join(str(x) for x in comb) + "_l" + str(lambda_)
            models["models"][model_name] = {
                "power_x":comb.tolist(),
                "lambda":lambda_,
                "name":model_name,
                "metrics_cv": [],
                "metrics_tr": [],
                "gradient":gradient,
                "optimizer":"basic",
                "accuracy":0,
                "theta":[[1 for _ in range(sum(comb) + 1)] for _ in range(number_of_label)],
                "total_it": 0
            }
    
    with open("models.yml", 'w') as outfile:
        yaml.dump(models, outfile, default_flow_style=None)
    print(f"{colors.green}Models successfuly initialize, saved in models.yml")
    return models


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


def format(arr: np.ndarray, label: int):
    """
    get an array and a label value, return a copy of array where
    label value in it is equal to 1
    and value different of label is equal to 0
    """
    copy = arr.copy()
    copy[:, 0][copy[:, 0] != label] = -1
    copy[:, 0][copy[:, 0] == label] = 1
    copy[:, 0][copy[:, 0] == -1] = 0
    return copy


def one_vs_all(k_folds, alpha: float, max_iter: int, model: dict, nb_of_label: int):
    """
    k_folds => list of numpy array x_train, y_train, x_test, y_test
    alpha => learning rate, max_iter => max iteration
    model => list of models
    nb_of_label => the number of labels to predict
    This function predict one label vs all of the other (binary classification)
    then "concat" all of this binary classification to create a nb_of_label classification
    it returns list of thetas/weights after training, evolution of evaluation metrics on the training and cross validation set (cv)
    and the accuracy score of the model (last metrics of cv set)
    """
    x_train, y_train, x_test, y_test = normalize(k_folds) # normalize test and train sets
    y_hat_all = pd.DataFrame() # dataframe to store prediction of each label

    metrics_cv, metrics_tr = np.zeros((max_iter, )), np.zeros((max_iter, )) # save historic metrics of each label training, get the mean at the end
    res_theta = [] # store finals theta for each binary classification

    for label in range(nb_of_label):
        # format y [house1, house2, house3 etc..] into "binarize" label: format(y, house1) => [1, 0, 0]
        binary_y_train, binary_y_test = format(y_train, label), format(y_test, label) 
        theta = np.array(model["theta"][label]).reshape(-1, 1) # get thetas from models

        # create logistic regression
        my_lr = LR(theta, alpha=alpha, max_iter=max_iter, lambda_=float(model["lambda"]), gradient=model["gradient"], optimizer=model["optimizer"])
        # fit return the metrics on training set and cross validation set, here the metrics is the accuracy
        tmp_metrics_tr, tmp_metrics_cv = my_lr.fit_(x_train, binary_y_train, x_test, binary_y_test, fct_metrics=accuracy_score_)

        # add it to the save
        metrics_tr = np.add(metrics_tr, tmp_metrics_tr)
        metrics_cv = np.add(metrics_cv, tmp_metrics_cv)

        # predict y_test depending on x_test, y_hat_one is binary prediction
        y_hat_one = my_lr.predict_(x_test)
        # add it to y_hat_all
        y_hat_all[label] = y_hat_one.reshape(len(y_hat_one))
        # add theta
        res_theta.append([float(tta) for tta in my_lr.theta])

    # compute mean of metrics for each training on label
    metrics_cv, metrics_tr = metrics_cv / nb_of_label, metrics_tr / nb_of_label
    return res_theta, metrics_cv, metrics_tr, float(metrics_cv[-1])


def normalize(k_folds: tuple):
    """
    get k_folds in argument (x_train, y_train, x_test, y_test)
    and return x_train, y_train, x_test, y_test but normalized and labelized if needed
    Normalize trains et and test set separately so model cannot have information of test set via normalization
    """
    x_train, y_train, x_test, y_test = k_folds
    x_train, y_train = Normalizer(x_train, norm="minmax").normalize().astype(float), Normalizer(y_train, norm="minmax").normalize().astype(float)
    x_test, y_test = Normalizer(x_test, norm="minmax").normalize().astype(float), Normalizer(y_test, norm="minmax").normalize().astype(float)
    return x_train, y_train, x_test, y_test


def train_all(yml_file: dict, X:np.ndarray, Y:np.ndarray, alpha: float, max_iter: int):
    """
    yml_file where all models are stored
    alpha the learning rate

    train all model on cross validation set
    """
    nb_of_label = len(np.unique(Y))  # get number of label to predict
    k = 4 # define K for k_folds algorithm

    #iterate over models
    for models_name in tqdm(yml_file["models"], leave=False):
        model = yml_file["models"][models_name]  # get model data
        X_poly = add_polynomial_features(X, model["power_x"])  # add corresponding polynom to features
        mean_metrics_cv, mean_metrics_tr = np.zeros((max_iter, )), np.zeros((max_iter, ))  # th mean of metrics is necessary because of the 1vsAll and cross validation
        model["accuracy"] = 0  # reset accuracy
        # iterate over k_folds
        for k_folds in tqdm(cross_validation(X_poly, Y, k), leave=False):
            # one_vs_all return the models theta, and metrics
            theta, metrics_cv, metrics_tr, accuracy = one_vs_all(k_folds, alpha, max_iter, model, nb_of_label)
            # store metrics to get the mean of it
            mean_metrics_cv, mean_metrics_tr = np.add(mean_metrics_cv, metrics_cv), np.add(mean_metrics_tr, metrics_tr)

            # add accuracy to get the mean of it
            model["accuracy"] = model["accuracy"] + accuracy
        
        # compute mean of metrics and accuracy
        mean_metrics_cv, mean_metrics_tr = mean_metrics_cv / k, mean_metrics_tr / k
        model["accuracy"] = model["accuracy"] / k

        # append to model previous metrics
        model["metrics_cv"] += mean_metrics_cv.tolist()
        model["metrics_tr"] += mean_metrics_tr.tolist()
        model["theta"] = theta
        model["total_it"] += max_iter

    # find the model with the best accuracy
    accuracy_list = np.array([[str(key), model["accuracy"]] for key, model in yml_file["models"].items()])
    yml_file["data"]["best_model"] = str(accuracy_list[accuracy_list[:, 1].astype('float64').argmax()][0])
    with open("models.yml", 'w') as outfile:
        yaml.dump(yml_file, outfile, default_flow_style=None)
    print(f"{colors.green}All models train, saved in models.yml")


def display_all(yml_file: dict):
    size = math.ceil(math.sqrt(yml_file["data"]["number_of_models"]))
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(16, 8))
    fig.tight_layout()
    idx = 0
    for model_name, model in yml_file["models"].items():
        dict_ = {
            "iteration":range(model["total_it"]),
            "metrics_cv":model["metrics_cv"],
            "metrics_tr":model["metrics_tr"]
        }
        data = pd.DataFrame(dict_)
        sns.lineplot(x='iteration', y='value', hue='variable', data=pd.melt(data, ['iteration']), ax=axs[idx % size][math.floor(idx / size)], legend=True if idx == 0 else False).set(title=model_name)
        idx += 1
    plt.show()


def test_gradient(X:np.ndarray, Y:np.ndarray, alpha: float, max_iter: int):
    nb_of_label = len(np.unique(Y))  # get number of label to predict
    model = {
            "power_x":[1 for _ in range(X.shape[1])],
            "lambda":0,
            "name":"test_gradient",
            "metrics_cv": [],
            "metrics_tr": [],
            "gradient":"batch",
            "optimizer":"basic",
            "accuracy":0,
            "theta":[[1 for _ in range(X.shape[1] + 1)] for _ in range(nb_of_label)],
            "total_it": 0
    }
    gradients = {
        "it":range(max_iter)
    }
    k_folds = data_spliter(X, Y, 0.9)
    for gradient in ["batch", "mini_batch", "stohastic"]:
        time_ = []
        for opt in ["basic", "adam"]:
            model["gradient"] = gradient
            model["optimizer"] = opt
            start_time = time.time()
            theta, metrics_cv, metrics_tr, accuracy = one_vs_all(k_folds, alpha, max_iter, model, nb_of_label)
            time_.append(f"{gradient}_{opt} {(time.time() - start_time):.3f}s")
            print(f"{colors.green}model with {gradient}_{opt} compute in: {(time.time() - start_time):.2f}s")
            gradients[f"{gradient}_{opt}"] = metrics_tr
        plt.figure()
        sns.lineplot(x='it', y='value', hue='variable', data=pd.melt(pd.DataFrame(gradients), ['it'])).set(title=time_)
        gradients = {
            "it":range(max_iter)
        }
    plt.show()


def parse(data: pd.DataFrame):
    """
    clean data dans return X, Y
    """
    data = cleaner(data, verbose=False)  # replace nan from data
    X = Normalizer(data[["Astronomy", "Best Hand","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes",
        "History of Magic","Transfiguration","Potions","Charms","Flying"]].to_numpy()).labelize()
    Y = data[["Hogwarts House"]]
    return X, Y


def main(argv):
    print(f"{colors.green}\
        USAGE:\n\t\
        python3 logreg_train.py [-f | --file] [path_to_dataset] [-l] [-i] [-g] [--train] [--display] [--reset]\n\t\
        -l : float needed, learning rate of models.\n\t\
        -i : int needed, maximum iteration.\n\t\
        --train : will train all models.\n\t\
        --display : display metrics curve of each models.\n\t\
        --gradient : test different gradient method.\n\t\
        -g : set gradient method on reset [batch, mini_batch, stohastic].\n\t\
    ")

    try:
        opts, args = getopt.getopt(argv, "f:l:i:g:", ["file=", "reset", "train", "display", "gradient"])
    except getopt.GetoptError as inst:
        error(inst)

    data = None
    yml_file = load_yml_file("models.yml")
    learning_rate, max_iter = 0.3, 250
    gradient = "batch"
    file_path = None

    try:
        for opt, arg in opts:
            if opt in ["-f", "--file"]:
                data = load_data(arg)
                file_path = arg
            elif opt in ["-l"]:
                learning_rate = float(arg)
            elif opt in ["-i"]:
                max_iter = int(arg)
            elif opt in ["-g"]:
                gradient = str(arg)
    except Exception as inst:
        error(inst)

    if data is None:
        error("No data file provided.")

    try:
        X, Y = parse(data)
        for opt, arg in opts:
            if opt in ["--reset"]:
                reset(range(1, 11), range(0, 1), X, Y, file_path, gradient)
            elif opt in ["--train"]:
                train_all(yml_file, X, Y, alpha=learning_rate, max_iter=max_iter)
            elif opt in ["--display"]:
                display_all(yml_file)
            elif opt in ["--gradient"]:
                test_gradient(X, Y, alpha=learning_rate, max_iter=max_iter)
    except Exception as inst:
        raise inst
        # error(inst)


if __name__ == "__main__":
    main(sys.argv[1:])