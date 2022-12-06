import numpy as np
import yaml

from utils.Normalizer import Normalizer
from utils.mylinearregression import MyLinearRegression as MyLR
from utils.common import colors


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


def predict_features(data, target_feature='All', verbose=False):

    if target_feature != 'All' and target_feature not in tab_feature:
        print(f"Error : {colors.red}{target_feature}{colors.reset} is not a valid feature.")
        print("Available columns : ")
        for feat in tab_feature:
            print(f"\t{colors.green}{feat}{colors.reset}")    
        return

    nb_predict = 0
    data_return = data.copy()
    for li, line in enumerate(data.index):
        for idx,col in enumerate(data.columns):
            if str(data[col][line]) == 'nan' and (target_feature == 'All' or col==target_feature) and col not in ['Hogwarts House']:
                list_feature = []
                for feat in tab_feature:
                    if feat not in ['Defense Against the Dark Arts', col]:
                        list_feature.append(feat)
                Xs = data[list_feature].values
                if verbose:
                    print(f"for #{colors.yellow}{data['Index'][line]}{colors.reset} {colors.green}{data['First Name'][line]} {data['Last Name'][line]}{colors.reset}: {colors.blue}{col}{colors.reset}[{idx}] to predict ... ", end='')
                model = load_yaml(col)
                scaler_x = Normalizer()
                scaler_y = Normalizer()
                if model is None:
                    if verbose:
                        print(f"{colors.red}No training model.{colors.reset}")
                    break
                else:
                    scaler_x.mean_ = np.array(model['mean_x'])
                    scaler_y.mean_ = model['mean_y']
                    scaler_x.std_ = np.array(model['std_x'])
                    scaler_y.std_ = model['std_y']
                    thetas = [float(theta) for theta in model['thetas']]
                    xx = scaler_x.norme(Xs)[li].reshape(1,-1)
                    mylr = MyLR(thetas=thetas)
                    yy = mylr.predict_(xx)
                    yy = scaler_y.inverse(yy)
                    if verbose:
                        print(data[col][line], yy[0])
                    data_return.at[line, col] = float(yy[0])
                    nb_predict += 1
                    break
    if verbose:
        print(f"There have been {colors.green}{nb_predict}{colors.reset} update predictions.")
    return data_return
                

# def main(argv):
#     try:
#         opts, args = getopt.getopt(argv, "f:", ["feature=","help"])
#     except getopt.GetoptError as inst:
#         print(f"ici:{inst}")
#         sys.exit(2)
#     features = 'All'
#     for opt, arg in opts:
#         if opt in ["-h", "--help"]:
#             help_me()
#             return
#         if opt in ["-f", "--feature"]:
#             features = arg
#     print("Predict_feature starting ...")
#     try:
#         # Importation of the dataset
#         data = pd.read_csv("datasets/dataset_train.csv")
#         data = clean(data)
#     except:
#         print("Issue when trying to retrieve the dataset.", file=sys.stderr)
#         sys.exit()
#     main_loop(data=data, target_feature=features)
#     print("Good by !")

# def help_me():
#     print("predic_feature scans the datasets/data_trains.csv file, \nand for each missing value (Nan) predicts with the corresponding model.")
#     print("options:")
#     print(f"\t[-f, --feature] FEATURE : trains on a requested feature")
#     print("list of available features:")
#     print("\tArithmancy, Astronomy, Herbology, Divination, Muggle Studies, Ancient Runes,")
#     print("\tHistory of Magic, Transfiguration, Potions, Care of Magical Creatures, Charms, Flying")

# def predict_features(data, verbose = False):
#     main_loop(data, 'All', verbose=verbose)

# if __name__ == "__main__":
#     main(sys.argv[1:])