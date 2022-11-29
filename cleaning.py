

from cmath import nan
import getopt
import sys

def load_data(path: str):
    """ load .csv file with path and return Dataframe of the dataset and header of the dataset droped by inused columns"""
    with open(path, "r") as stream:
        try:
            data = pd.read_csv(stream)
        except Exception as inst:
            print(inst)
            sys.exit(2)
    # data.drop(['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday'], axis = 1, inplace = True)
    return data, data.columns

import pandas as pd

def main(argv):
    file = None
    try:
        opts, _ = getopt.getopt(argv, "f:e:a:", ["file=", "feature=", "all"])
    except getopt.GetoptError as inst:
        print(inst)
        sys.exit(2)
    
    for opt, arg in opts:
        print(opt,arg)
        if opt in ["-f", "--file"]:
            file = arg
    data, _ = load_data(file)
    data = cleaning(data, nb_nan_lim=1, verbose=True)

def cleaning(data, nb_nan_lim = 1, verbose = False):
    """ function to clean dataset
        it deletes the lines containing more than <nb_nam_lim> values 'Nan'
        args:
            data :  Dataframe
            nb_nan_lim: int, if strictly greater than nb_nan_lim, we delete the
            verbose: False no print, True some print info
        return:
            a copy of the DataFrame data
    """
    if verbose:
        print(f'len before cleaning = {len(data)}')
    nb_del = 0
    if verbose:
        print(f" <<Nan>> Value found in dataset : ")
        print(data.isnull().sum())
    to_del = []
    for line in data.index: 
        nb_nan = 0
        for col in data.columns:
            if str(data[col][line]) == 'nan':
                nb_nan += 1
        if nb_nan > nb_nan_lim:
            to_del.append(int(data['Index'][line]))
            nb_del += 1
    update_data = data.drop(to_del)
    if verbose:
        print(f"{nb_del} deleted. -> new len = {len(update_data)}.")
    return update_data

if __name__ == "__main__":
    main(sys.argv[1:])