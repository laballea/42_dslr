import numpy as np
import getopt, sys
import pandas as pd
import matplotlib.pyplot as plt

from utils.logisticregression import LogisticRegression as LR
from utils.normalizer import Normalizer
from utils.utils_ml import add_polynomial_features
from utils.cleaning import cleaner
from utils.metrics import accuracy_score_
from utils.common import load_data, load_yml_file, error, colors
from utils.confusion_matrix import confusion_matrix_


def format_all(arr: np.ndarray):
    """
    get an array of dimension (m, number of label) in argument representing probability of each label
    return an array of dimension (m, 1), where the best probability is choosen
    """
    result = []
    for index, row in arr.iterrows():
        result.append(row.idxmax())
    result = np.array(result).reshape(-1, 1)
    return result


def one_vs_all_predict(X: np.ndarray, model: dict, nb_of_label: int):
    """
    depending on X matrix, predict Y
    """
    y_hat_all = pd.DataFrame() # dataframe to store prediction of each label

    X_poly = add_polynomial_features(X, model["power_x"]) # add polynome to X

    # predict for each label
    for label in range(nb_of_label):
        theta = np.array(model["theta"][label]).reshape(-1, 1) # get thetas of saved best model

        # create Logistic Regression class
        my_lr = LR(theta, lambda_=float(model["lambda"]))

        # predict y_test depending on x_test, y_hat_one is binary prediction
        y_hat_one = my_lr.predict_(X_poly)
        # add it to y_hat_all
        y_hat_all[label] = y_hat_one.reshape(len(y_hat_one))

    return format_all(y_hat_all)


def predict(yml_file: dict, dataTest: pd.DataFrame, dataTrain: pd.DataFrame):
    """
    predict dataTest X values via the best model on yml_file, use dataTrain to normalize value of dataTest
    """
    # get best model
    model = yml_file["models"][yml_file["data"]["best_model"]]

    # parse train and test set
    X_train, Y_train = parse(dataTrain)
    X_test, Y_test = parse(dataTest)

    # get the number of label that we can predict
    nb_of_label = len(np.unique(Y_train))

    # create normalizer class
    normizer_X = Normalizer(X_train.to_numpy(), norm="minmax")
    normizer_Y = Normalizer(Y_train.to_numpy(), norm="minmax")

    # normalize and labelize x_test
    X_test_n = normizer_X.normalize(X_test.to_numpy()).astype(float)

    # predict y_test via x_test
    y_hat = one_vs_all_predict(X_test_n, model, nb_of_label=nb_of_label)

    # denormalize y_hat
    y_hat = normizer_Y.denormalize(y_hat)

    # create dict to match format requested on subject
    result = {
        "Index": range(len(y_hat)),
        "Hogwarts House":y_hat.reshape(len(y_hat),)
    }
    # write prediction on csv format in house.csv
    pd.DataFrame(data=result).to_csv("houses.csv", index=False)
    print(f"{colors.green}File houses.csv successfuly create !")


def parse(data: pd.DataFrame):
    """
    replace all nan by a prediction or mean value via cleaner
    return X and Y as Dataframe
    """
    data = cleaner(data, verbose=False)  # replace nan from data
    X = data[["Astronomy", "Best Hand","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes",
        "History of Magic","Transfiguration","Potions","Charms","Flying"]]
    Y = data[["Hogwarts House"]]
    return X, Y


def verify():
    predict = Normalizer(load_data("houses.csv", type_data='houses')[["Hogwarts House"]].to_numpy()).labelize()
    truth = Normalizer(load_data("datasets/data_truth.csv", type_data='houses')[["Hogwarts House"]].to_numpy()).labelize()
    print(f"{colors.green}{accuracy_score_(predict, truth)}% accuracy")
    conf_matrix = confusion_matrix_(truth, predict, df_option=False)
    _, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='12')
    plt.xlabel('Predictions', fontsize=12)
    plt.ylabel('True values', fontsize=12)
    plt.title('Confusion Matrix', fontsize=12)
    plt.show()

def main(argv):
    print(f"{colors.green}USAGE:\n\tpython3 logreg_predict.py [-f | --file] [path_to_dataset] [-v | --verify]\n\t-v | --verify : verify prediction.")
    try:
        opts, args = getopt.getopt(argv, "f:v", ["file=", "verify"])
    except getopt.GetoptError as inst:
        error(inst)


    data_test = None
    yml_file = load_yml_file("models.yml")
    data_train = load_data(yml_file["data"]["data_train_path"], type_data='train')

    for opt, arg in opts:
        if opt in ["-f", "--file"]:
            data_test = load_data(arg, type_data='test')

    if data_test is None:
        error("No test file provided.")

    try:
        predict(yml_file, data_test, data_train)
        for opt, arg in opts:
            if opt in ["-v", "--verify"]:
                verify()
    except Exception as inst:
        error(inst)

if __name__ == "__main__":
    main(sys.argv[1:])