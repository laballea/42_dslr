import getopt
import sys
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt

from cleaning import clean
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from Normalizer import Normalizer
from mylinearregression import MyLinearRegression as MyLR
from common import colors


def save_model(file, thetas, scaler_x, scaler_y):

    print(f"Saving model '{colors.yellow}{file}{colors.reset}'... ", end='')
    model = {}
    model['thetas'] = [float(theta) for theta in thetas]        
    model['mean_x'] = [float(x) for x in scaler_x.mean_]
    model['mean_y'] = float(scaler_y.mean_)
    model['std_x'] = [float(std) for std in scaler_x.std_]
    model['std_y'] = float(scaler_y.std_)
    print(f"{colors.green}Ok{colors.reset}")

    with open(file, 'w') as outfile:
        yaml.dump(model, outfile, sort_keys=False, default_flow_style=None )


def load_rules():
    #load models rules
    rules = []
    try:
        with open('models/myrl_rules.yaml') as infile:
            models_rules = yaml.safe_load(infile)
            rules = models_rules
            return rules
    except IOError:
        print("Error no 'models/mylr_rules.yaml'.")
        sys.exit()


def search_model(data, target_feature, graph=False, save=True):
    
    rules = load_rules()
    file = "models/"+target_feature.replace(" ", "_")+".yaml"
    print(f"creation of  {colors.green}{file}{colors.reset}...")
    target = data[target_feature].values.reshape(-1, 1)
    Xs = data.drop(target_feature, axis=1).to_numpy()
    nb_features = Xs.shape[1]
    # split dataset
    x_train, x_test, y_train, y_test = data_spliter(Xs, target.reshape(-1,1), 0.8)

    x_test_to_plot = x_test
    y_test_to_plot = y_test

    #normalisation

    scaler_x = Normalizer(x_train)
    scaler_y = Normalizer(y_train)

    #zscore
    x = scaler_x.norme(x_train)
    y = scaler_y.norme(y_train)
    #minmax
    # x = np.array(x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    # y = np.array(y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))

    #Zscore
    x_test = scaler_x.norme(x_test)
    y_test = scaler_y.norme(y_test)

    #minmax
    # x_test = np.array(x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
    # y_test = np.array(y_test - np.min(y_test)) / (np.max(y_test) - np.min(y_test))

    hypo = [1 for _ in range(nb_features)]
    thetas = [1 for _ in range(nb_features + 1)]
    if target_feature in rules.keys():
        alpha = rules[target_feature]['alpha']
        iter = rules[target_feature]['iter']
    else:
        alpha = rules['default']['alpha']
        iter = rules['default']['iter']
    x_ = add_polynomial_features(x, hypo)
    x_test_ = add_polynomial_features(x_test, hypo)
    print(f"Regression with {colors.yellow}alpha={colors.reset}{colors.blue}{alpha}{colors.reset} and {colors.yellow}max_iter={colors.reset}{colors.reset}{colors.blue}{iter}{colors.reset}")
    mylr = MyLR(thetas, alpha=alpha, max_iter=iter, progress_bar=True)
    
    mse_list = mylr.fit_(x_, y)
    y_hat_normalise =  mylr.predict_(x_test_)
    y_hat = scaler_y.inverse(y_hat_normalise)
    mse = MyLR.mse_(y_test, mylr.predict_(x_test_))
    mse_training = MyLR.mse_(y, mylr.predict_(x_))
    print(f"\tMSE test = {colors.green}{mse}{colors.reset}")
    print(f"\tMSE training = {colors.green}{mse_training}{colors.reset}")
    # print(f"\tdiff = {colors.blue}{mse_training - mse}{colors.reset}")

    if save:
        # save model
        save_model(file, mylr.thetas, scaler_x, scaler_y)
    else:
        print(f"Model for {colors.yellow}{target_feature}{colors.reset}... {colors.red}Not saved{colors.reset}")

    if graph:
        plt.figure()
        plt.scatter(np.arange(len(x_test_to_plot)), y_test_to_plot, c='b', marker='o', label=target_feature)
        plt.scatter(np.arange(len(x_test_to_plot)), y_hat, c='r', marker='x', label="predicted")
        plt.title(f"Feature : {target_feature} - MSE={mse:0.2e}")
        plt.legend()

        plt.figure()
        plt.plot(np.arange(mylr.max_iter), mse_list)
        plt.title(f"Feature : {target_feature} - MSE={mse:0.2e}")
        plt.show()


def main_loop(target_feature, save=False, graph=False):
    try:
        # Importation of the dataset
        data = pd.read_csv("datasets/dataset_train.csv")
       
        # remove of raw with Nan values > 2
        data = clean(data, verbose=True)
        data = data.dropna()
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()
    data.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Defense Against the Dark Arts'], axis=1, inplace=True)
    
    if target_feature == 'All':
        for target_feature in data.columns:
            search_model(data, target_feature, graph=graph, save=save)
    else:
        if target_feature in data.columns:
            search_model(data, target_feature, graph=graph, save=save)
        else:
            print(f"Error : {colors.red}{target_feature}{colors.reset} is not a valid feature.")
            print("Available columns : ")
            for feat in data.columns:
                print(f"\t{colors.green}{feat}{colors.reset}")


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:sgh", ["feature=", "save","graph","help"])
    except getopt.GetoptError as inst:
        print(f"ici:{inst}")
        sys.exit(2)
    graph = False
    save = False
    features = 'All'
    for opt, arg in opts:
        if opt in ["-h", "--help"]:
            help_me()
            return
        if opt in ["-g", "--graph"]:
            graph = True
        if opt in ["-s", "--save"]:
            save = True
        if opt in ["-f", "--feature"]:
            features = arg
    print("Benchmar starting ...")
    main_loop(target_feature=features, save=save, graph=graph)
    print("Good by !")

def help_me():
    print("The Benchmark program, trains the models from the datasets/data_train.csv file and saves them in the models/ directory by feature name (ex Potions.yaml")
    print("options:")
    print(f"\t[-f, --feature] FEATURE : trains on a requested feature")
    print(f"\t[-s, --save] : save each model (by default it don't save)")
    print(f"\t[-g, -graph]: display the graph of predictions and the evolution of the MSE")
    print("list of available features:")
    print("\tArithmancy, Astronomy, Herbology, Divination, Muggle Studies, Ancient Runes,")
    print("\tHistory of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms, Flying")

if __name__ == "__main__":
    main(sys.argv[1:])