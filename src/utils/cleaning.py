import pandas as pd
import numpy as np
from cmath import nan
import getopt
import sys
from predict_features import predict_features

green = '\033[92m' # vert
blue = '\033[94m' # blue
yellow = '\033[93m' # jaune
red = '\033[91m' # rouge
reset = '\033[0m' #gris, couleur normale

def load_data(path: str):
    """ load .csv file with path and return Dataframe of the dataset and header of the dataset droped by inused columns"""
    with open(path, "r") as stream:
        try:
            data = pd.read_csv(stream)
        except Exception as inst:
            print(inst)
            sys.exit(2)
    return data, data.columns

def mean_(x: np.ndarray):
        """
            computes the mean of a given non-empty list or array x, using a for-loop.
            The method returns the mean as a float, otherwise None if x is an empty list or
            array.
        """
        # try:
        sum = 0
        for i in x:
            if str(i) != 'nan':
                sum += i
        return sum / len(x)

def compute_astronomy(row):
    """ compute the Astronomy Value with 'Defense Ag Dark Arts' Value"""
    if str(row["Astronomy"]) == 'nan':
        defense = row["Defense Against the Dark Arts"]
        return ((defense * 100) * -1)
    return row["Astronomy"]

def compute_defense(row):
    """ compute the Defense Against the Dark Arts Value with 'Astronomy' Value"""
    if str(row["Defense Against the Dark Arts"]) == 'nan':
        astronomy = row["Astronomy"]
        return ((astronomy / 100) * -1)
    return row["Defense Against the Dark Arts"]

def clean(data, nb_nan_lim = 2, verbose = False):
    """ function to clean dataset
        it deletes the lines containing more than <nb_nam_lim> values 'Nan'
        and put the mean of the column  the first value 'nan' if there is more than nb_nan_lim
        args:
            data :  Dataframe
            nb_nan_lim: int, if strictly greater than nb_nan_lim, we delete the
            verbose: False no print, True some print info
        return:
            a copy of the DataFrame data clean
    """
    tab_feature = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes',
   'History of Magic', 'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']

    if verbose:
        print("compute Astronomy...", end='')
    data["Astronomy"]=data.apply(lambda x : compute_astronomy(x), axis=1)
    if verbose:
        print("ok")
        print("compute Defense...", end='')
    data["Defense Against the Dark Arts"]=data.apply(lambda x : compute_defense(x), axis=1)
    if verbose:
        print("ok")
        print(f'len before cleaning = {len(data)}')
        print(f" <<Nan>> Value found in dataset : ")
        print(data.isnull().sum())
        print(f"\t\tTotal = {data.isnull().sum().sum()}")
    #compute nb of raw to del and raw to mean
    nb_del = 0
    nb_mean = 0
    to_del = []
    to_mean = []
    for line in data.index: 
        nb_nan = 0
        for col in data.columns:
            if str(data[col][line]) == 'nan':
                nb_nan += 1
        if nb_nan > nb_nan_lim:
            to_del.append(int(data['Index'][line]))
            nb_del += 1
        elif nb_nan == nb_nan_lim:
            to_mean.append(int(data['Index'][line]))
            nb_mean += 1
    
    update_data = data.copy()
    #mean raw
    for line in to_mean:
        for col in tab_feature:
            m_ = mean_(np.array(data[col]))
            if str(data[col][line]) == 'nan':
                update_data.at[line, col] = m_
                break
    #delete raw        
    update_data.drop(to_del)
    if verbose:
        print(f"{nb_del} deleted. -> new len = {len(update_data)}.")
        print(f"nb of values compute with mean of columns {nb_mean}")
    # mean if 
    return update_data

def cleaner(data, verbose=False):
    return (predict_features(clean(data=data, verbose=verbose), target_feature='All', verbose=verbose))

def main(argv):
    file = None
    try:
        opts, _ = getopt.getopt(argv, "f:v", ["file=", "verbose"])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    verbose = False
    for opt, arg in opts:
        if opt in ["-v", "--verbose"]:
            verbose = True
        if opt in ["-f", "--file"]:
            file = arg
    if file is None:
        print(f"USAGE:\n\t$>python3 cleaning.pyp -f DATASET.csv [-v]\n\t-v : verbose mode.")
        return
    data, _ = load_data(file)
    before = data.isnull().sum().sum()
    data = cleaner(data, verbose=verbose)
    if verbose:
        print(f"\n     \tBefore the cleaner's functions there wase {red}{before}{reset} 'Nan' Values *****")
        print(f"\n*****\tAfter the cleaner's functions there is {green}{data.isnull().sum().sum()}{reset} 'Nan' Value")

if __name__ == "__main__":
    print("cleaning programme...")
    main(sys.argv[1:])
    print("Good bye.")