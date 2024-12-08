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

def visualize_genre_distribution(df_movies, values=False, save=False, save_path='../output/genre_distribution.png', title='Distribution des films par genre'):
    """
    Visualize the distribution of movies per genre

    Parameters
    ----------
        df_movies : pandas.DataFrame, the data from the movies csv file
        values : bool, whether to display the values on the bars
        save : bool, whether to save the plot
        save_path : str, the path to save the plot
        title : str, the title of the plot
    """

    genre_counts = process_movies(df_movies)
    plt.figure(figsize=(12, 8))
    ax = genre_counts.plot(kind='bar', color='orange', zorder=3)
    plt.title(title)
    plt.xlabel('Genre')
    plt.ylabel('Nombre de films')
    plt.xticks(rotation=45)
    plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.3, zorder=0)
    if values:
        for i in range(len(genre_counts)):
            ax.text(i, genre_counts[i], genre_counts[i], ha='center', va='bottom')

    if save:
        plt.savefig(save_path)

    plt.show()


def clean_and_save_data(df_movies, df_ratings):
    """
    Clean the data and save it to new csv files

    Parameters
    ----------
        df_movies : pandas.DataFrame, the data from the movies csv file
        df_ratings : pandas.DataFrame, the data from the ratings csv file
    """
    
    df_movies = df_movies[df_movies['genres'] != '(no genres listed)']
    listed_movie_ids = df_movies['movieId'].unique()
    df_ratings = df_ratings[df_ratings['movieId'].isin(listed_movie_ids)]

    # Adjusting rating values
    df_ratings.loc[df_ratings['rating'] == 5.5, 'rating'] = 5
    df_ratings.loc[df_ratings['rating'] == 4.5, 'rating'] = 4
    df_ratings.loc[df_ratings['rating'] == 3.5, 'rating'] = 3
    df_ratings.loc[df_ratings['rating'] == 2.5, 'rating'] = 2
    df_ratings.loc[df_ratings['rating'] == 1.5, 'rating'] = 1
    df_ratings.loc[df_ratings['rating'] == 0.5, 'rating'] = 1

    # save the cleaned data
    df_movies.to_csv('../data/movies1.csv', index=False)
    df_ratings.to_csv('../data/ratings1.csv', index=False)

    print("Les fichiers movies1.csv et ratings1.csv ont été créés avec succès.")