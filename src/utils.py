# Description: This file contains the functions to extract the data from the csv files and the functions to plot the data.

# =================================================================================
# LIBRAIRIES
# ================================================================================= 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA



# =================================================================================
# FUNCTIONS
# =================================================================================

#__________________________________________________________________________________
# Functions to extract the data


def init_traitement_DB(data_path) :
    """
    Load the data from the csv file and remove the first column

    Parameters
    ----------
        data_path : str, the path to the csv file

    Output
    ------
        df : pandas.DataFrame, the data from the csv
    """
    #Lecture du fichier
    df = pd.read_csv(data_path)
    
    # Suppression de la première colonne inutile
    df = df.iloc[:,1:]

    return df
