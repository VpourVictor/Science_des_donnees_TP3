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
from collections import defaultdict



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

    return df


#__________________________________________________________________________________
# Function first question

def process_movies(df_movies):
    """
    Process the data from the movies csv file to get the number of movies per genre

    Parameters
    ----------
        df_movies : pandas.DataFrame, the data from the movies csv file

    Output
    ------
        genre_counts : pandas.Series, the number of movies per genre
    """

    # Ignore movies without genre
    df_movies = df_movies[df_movies['genres'] != '(no genres listed)']

    # Initialiser un dictionnaire pour compter les genres
    genre_counts = defaultdict(int)

    # Explore the dataframe
    for genres in df_movies['genres']:
        for genre in genres.split('|'):
            genre_counts[genre] += 1

    genre_counts = pd.Series(genre_counts)

    return genre_counts

def visualize_genre_distribution(df_movies, values=False, save=False, save_path='../output/genre_distribution.png'):
    """
    Visualize the distribution of movies per genre

    Parameters
    ----------
        df_movies : pandas.DataFrame, the data from the movies csv file
        values : bool, whether to display the values on the bars
        save : bool, whether to save the plot
        save_path : str, the path to save the plot
    """

    genre_counts = process_movies(df_movies)
    plt.figure(figsize=(12, 8))
    ax = genre_counts.plot(kind='bar', color='orange', zorder=3)
    plt.title('Nombre de films par genre')
    plt.xlabel('Genre')
    plt.ylabel('Nombre de films')
    plt.xticks(rotation=45)
    plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
    if values:
        for i in range(len(genre_counts)):
            ax.text(i, genre_counts[i], genre_counts[i], ha='center', va='bottom')

    if save :
        plt.savefig(save_path)

    plt.show()