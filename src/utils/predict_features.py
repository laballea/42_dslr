import numpy as np
import yaml

from utils.normalizer import Normalizer
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
                scaler_x = Normalizer(norm='zscore')
                scaler_y = Normalizer(norm='zscore')
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
                    xx = scaler_x.zscore(Xs)[li].reshape(1,-1)
                    mylr = MyLR(thetas=thetas)
                    yy = mylr.predict_(xx)
                    yy = scaler_y.unzscore(yy)
                    if verbose:
                        print(data[col][line], yy[0])
                    data_return.at[line, col] = float(yy[0])
                    nb_predict += 1
                    break
    if verbose:
        print(f"There have been {colors.green}{nb_predict}{colors.reset} update predictions.")
    return data_return
 