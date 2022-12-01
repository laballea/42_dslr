import getopt
import sys
from cleaning import cleaning, compute_astronomy
import pandas as pd
import numpy as np
import yaml
from Normalizer import Normalizer
from mylinearregression import MyLinearRegression as MyLR

green = '\033[92m' # vert
blue = '\033[94m' # blue
yellow = '\033[93m' # jaune
red = '\033[91m' # rouge
reset = '\033[0m' #gris, couleur normale

tab_feature = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes',
   'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

def load_yaml(feature):
    file = "models/"+feature.replace(" ", "_")+".yaml"
    try:
        with open(file) as infile:
            model = yaml.safe_load(infile)
            return model
    except IOError:
        return None

def main_loop(target_feature):

    try:
        # Importation of the dataset
        data = pd.read_csv("datasets/dataset_train.csv")
        # remove of raw with Nan values > 1
        data = cleaning(data, nb_nan_lim=1, verbose=True)

        print("compute Astronomy...", end='')
        data["Astronomy"]=data.apply(lambda x : compute_astronomy(x), axis=1)
        print("ok")
    except:
        print("Issue when trying to retrieve the dataset.", file=sys.stderr)
        sys.exit()

    
    nb_predict = 0
    data_return = data.copy()
    for li, line in enumerate(data.index):
        for idx,col in enumerate(data.columns):
            if str(data[col][line]) == 'nan' and (target_feature == 'All' or col==target_feature):
                list_feature = []
                for feat in tab_feature:
                    if feat not in ['Defense Against the Dark Arts', col]:
                        list_feature.append(feat)
                Xs = data[list_feature].values
                print(f"for #{yellow}{data['Index'][line]}{reset} {green}{data['First Name'][line]} {data['Last Name'][line]}{reset}: {blue}{col}{reset}[{idx}] to predict ... ", end='')
                model = load_yaml(col)
                scaler_x = Normalizer()
                scaler_y = Normalizer()
                theta = (0.0, 0.0)
                if model is None:
                    print(f"{red}No training model.{reset}")
                    break
                else:
                    scaler_x.mean_ = np.array(model['mean_x'])
                    scaler_y.mean_ = model['mean_y']
                    scaler_x.std_ = np.array(model['std_x'])
                    scaler_y.std_ = model['std_y']
                    thetas = [float(theta) for theta in model['thetas']]
                    xx = scaler_x.norme(Xs)[li].reshape(1,-1)
                    mylr = MyLR(thetas=thetas)
                    yy =mylr.predict_(xx)
                    print(data[col][line], yy[0])
                    data_return.at[line, col] = float(yy[0])
                    nb_predict += 1
                    break
    print(f"There have been {green}{nb_predict}{reset} update predictions.")
    cleaning(data_return, verbose=True)
    return data_return
                

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:", ["feature=","help"])
    except getopt.GetoptError as inst:
        print(f"ici:{inst}")
        sys.exit(2)
    features = 'All'
    for opt, arg in opts:
        if opt in ["-h", "--help"]:
            help_me()
            return
        if opt in ["-f", "--feature"]:
            features = arg
    print("Predict_feature starting ...")
    main_loop(target_feature=features)
    print("Good by !")

def help_me():
    print("predic_feature scans the datasets/data_trains.csv file, \nand for each missing value (Nan) predicts with the corresponding model.")
    print("options:")
    print(f"\t[-f, --feature] FEATURE : trains on a requested feature")
    print("list of available features:")
    print("\tArithmancy, Astronomy, Herbology, Divination, Muggle Studies, Ancient Runes,")
    print("\tHistory of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms, Flying")

if __name__ == "__main__":
    main(sys.argv[1:])