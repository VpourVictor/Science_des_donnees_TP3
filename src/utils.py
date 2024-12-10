# Description: This file contains the functions to extract the data from the csv files and the functions to plot the data.

# =================================================================================
# LIBRAIRIES
# ================================================================================= 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, mean_squared_error
from joblib import Parallel, delayed
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

def filter_dataframes_by_threshold(threshold, df_movies, df_ratings):
    """
    Filter the movies and ratings DataFrames by removing users who have rated fewer than the threshold number of movies.

    Parameters
    ----------
        threshold : int, the threshold number of movies a user must have rated
        df_movies : pandas.DataFrame, the movies DataFrame
        df_ratings : pandas.DataFrame, the ratings DataFrame

    Output
    ------
        df_movies_thresholded : pandas.DataFrame, the filtered movies DataFrame
        df_ratings_thresholded : pandas.DataFrame, the filtered ratings DataFrame
    """
    user_counts = df_ratings['userId'].value_counts()
    valid_users = user_counts[user_counts >= threshold].index
    df_ratings_thresholded = df_ratings[df_ratings['userId'].isin(valid_users)]
    valid_movie_ids = df_ratings_thresholded['movieId'].unique()
    df_movies_thresholded = df_movies[df_movies['movieId'].isin(valid_movie_ids)]

    # Adding the reduction percentage
    r_movies = len(df_movies_thresholded)/len(df_movies)
    r_ratings = len(df_ratings_thresholded)/len(df_ratings)

    # Print the number of lines deleted
    print(f"Reduction in movies: {100 * (1 - r_movies):.2f}%")
    print(f"Reduction in ratings: {100 * (1 - r_ratings):.2f}%")

    return df_movies_thresholded, df_ratings_thresholded


# Question 4
def create_db_content_movie(df_movies, save=False, save_path='../data/movie_matrix.csv'):

    genres_dict = {}

    # Explore the df
    for index, row in df_movies.iterrows():
        movie_id = row['movieId']
        genres = row['genres']
        genres = genres.split('|')

        # Explore the genres
        for genre in genres:
            if genre not in genres_dict:
                genres_dict[genre] = set()
            genres_dict[genre].add(movie_id)

    # Creation of the dataframe : matrix with zeroes and ones
    content_matrix = pd.DataFrame(0, index=df_movies['movieId'], columns=genres_dict.keys())

    # Fill the matrix using the dictionnary
    for genre, movie_ids in genres_dict.items():
        content_matrix.loc[list(movie_ids), genre] = 1

    #save the matrix as a .csv file
    if save :
        content_matrix.to_csv(save_path)
        print("Content matrix saved !")
    
    return content_matrix

def calculate_user_profile(user_id, df_ratings, content_matrix):
    """
    Calculate the profile for a single user based on their ratings and the content matrix.

    Parameters
    ----------
        user_id : int, the user ID
        df_ratings : pandas.DataFrame, the ratings DataFrame
        content_matrix : pandas.DataFrame, the binary content matrix

    Output
    ------
        user_profile : pandas.Series, the user profile
    """
    user_ratings = df_ratings[df_ratings['userId'] == user_id]
    user_profile = pd.Series(0, index=content_matrix.columns)

    # Compute the user profile
    for index, row in user_ratings.iterrows():
        movie_id = row['movieId']
        rating = row['rating']
        user_profile += content_matrix.loc[movie_id] * rating

    user_profile = user_profile / user_profile.sum()

    return user_profile

def calculate_user_profiles(df_ratings, content_matrix, n_jobs=-1):
    """
    Calculate the user profiles based on their ratings and the content matrix.

    Parameters
    ----------
        df_ratings : pandas.DataFrame, the ratings DataFrame
        content_matrix : pandas.DataFrame, the binary content matrix
        n_jobs : int, the number of jobs to run in parallel

    Output
    ------
        user_profiles : pandas.DataFrame, the user profiles matrix
    """
    print("Calculating user profiles...")
    user_ids = df_ratings['userId'].unique()

    # Parallelize the computation of user profiles
    user_profiles = Parallel(n_jobs=n_jobs)(
        delayed(calculate_user_profile)(user_id, df_ratings, content_matrix) for user_id in user_ids
    )

    user_profiles_df = pd.DataFrame(user_profiles, index=user_ids)

    return user_profiles_df

#Question 5
def get_rating(id_user, id_movie, df_ratings):
    user_ratings = df_ratings[df_ratings['userId'] == id_user]
    rating = user_ratings[user_ratings['movieId'] == id_movie]
    
    if len(rating["rating"]) == 0:
        return 0
    else:
        return float(rating['rating'])

def get_mean_rating(id_user, df_ratings):
    user_ratings = df_ratings[df_ratings['userId'] == id_user]
    if len(user_ratings) == 0:
        return 0
    else:
        return round(np.mean(user_ratings['rating']),3)

def movie_mean_rating(movie_id, df_ratings):
    movie_ratings = df_ratings[df_ratings['movieId'] == movie_id]
    if len(movie_ratings) == 0:
        print("movie was not rated")
        return 0
    else:
        return round(np.mean(movie_ratings['rating']),3)
    
def get_movies_rated(id_user, df_ratings):
    user_ratings = df_ratings[df_ratings['userId'] == id_user]
    if len(user_ratings) == 0:
        return []
    else :
        return user_ratings["movieId"].tolist()

def get_users(df_ratings):
    return df_ratings["userId"].unique().tolist()

def get_same_movies(user1, user2, df_ratings):
    movies_user1 = get_movies_rated(user1, df_ratings)
    movies_user2 = get_movies_rated(user2, df_ratings)
    intersection = list(set(movies_user1) & set(movies_user2))
    return intersection

def get_users_rating_both(movie1, movie2, df_ratings):
    users = get_users(df_ratings)
    print("users : ", users)
    list_users = []
    for user in users:
        movies = get_movies_rated(user, df_ratings)
        if (movie1 in movies) and (movie2 in movies):
            list_users.append(user)
    return list_users

def pearson_correlation_users(id1, id2, df_ratings):
    mean_rating1 = get_mean_rating(id1, df_ratings)
    mean_rating2 = get_mean_rating(id2, df_ratings)
    list_same_movies = get_same_movies(id1, id2, df_ratings)
    if len(list_same_movies) == 0:
        print("error : no shared movies")
        return 0
    numerator = 0
    denominator1 = 0
    denominator2 = 0
    for movie_id in list_same_movies:
        comp1 = (get_rating(id1, movie_id, df_ratings) - mean_rating1)
        comp2 = (get_rating(id2, movie_id, df_ratings) - mean_rating2)
        numerator += comp1*comp2
        denominator1 += comp1**2
        denominator2 += comp2**2
    denominator1 = sqrt(denominator1)
    denominator2 = sqrt(denominator2)
    return numerator/(denominator1*denominator2)

def pearson_correlation_movies(id1, id2, df_ratings):
    mean_rating1 = movie_mean_rating(id1, df_ratings)
    mean_rating2 = movie_mean_rating(id2, df_ratings)
    list_users = get_users_rating_both(id1, id2, df_ratings)
    if len(list_users) == 0:
        print("error : no user rated both")
        return 0
    numerator = 0
    denominator1 = 0
    denominator2 = 0
    for user_id in list_users:
        comp1 = (get_rating(user_id, id1, df_ratings) - mean_rating1)
        comp2 = (get_rating(user_id, id2, df_ratings) - mean_rating2)
        numerator += comp1*comp2
        denominator1 += comp1**2
        denominator2 += comp2**2
    denominator1 = sqrt(denominator1)
    denominator2 = sqrt(denominator2)
    return numerator/(denominator1*denominator2)

def create_cross_validation_datas(data_path, folder_path):
    """ Splits the ratings1.csv file into 5 files of the same length in order to do a cross validation"""
    df = init_traitement_DB(data_path)
    # Shuffle the DataFrame randomly
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the shuffled DataFrame into 5 random subsets
    random_subsets = np.array_split(df_shuffled, 5)
    for i in range(5):
        random_subsets[i].to_csv(folder_path + 'ratings_partie' + str(i+1) + '.csv', index=False)

def create_train_test_subsets(data_path, folder_path):
    # Diviser les données en 80 % entraînement et 20 % test
    df = init_traitement_DB(data_path)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv(folder_path + 'ratings_apprentissage.csv', index=False)
    test.to_csv(folder_path + 'ratings_evaluation.csv', index=False)

def error(Ypred, Yval):
    error = 0
    n = len(Ypred)
    for i in range(n):
        error += abs(Ypred[i] - Yval[i])
    error /= n
    return error



# Question 7


def determine_optimal_clusters_para(user_profiles, cluster_method='K_means', max_clusters=10, n_jobs=-1):
    """
    Determine the optimal number of clusters using the silhouette score.

    Parameters
    ----------
        user_profiles : pandas.DataFrame, the user profiles matrix
        cluster_method : str, the clustering method ('K_means' or 'spectral')
        max_clusters : int, the maximum number of clusters to consider
        n_jobs : int, the number of jobs to run in parallel

    Output
    ------
        optimal_clusters : int, the optimal number of clusters
    """
    # Store the silhouette scores for different cluster sizes
    silhouette_scores = Parallel(n_jobs=n_jobs)(
        delayed(calculate_silhouette_score_para)(user_profiles, n_clusters, cluster_method)
        for n_clusters in range(2, max_clusters + 1)
    )

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"Optimal number of clusters is {optimal_clusters}")
    return optimal_clusters

def calculate_silhouette_score_para(user_profiles, n_clusters, cluster_method):
    """
    Calculate the silhouette score for a given number of clusters.

    Parameters
    ----------
        user_profiles : pandas.DataFrame, the user profiles matrix
        n_clusters : int, the number of clusters
        cluster_method : str, the clustering method ('K_means' or 'spectral')

    Output
    ------
        silhouette_avg : float, the average silhouette score
    """
    if cluster_method == 'K_means':
        # Train with kmeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(user_profiles)
    else:
        # Train with spectral clustering
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
        cluster_labels = spectral.fit_predict(user_profiles)

    # Compute the silhouette score
    silhouette_avg = silhouette_score(user_profiles, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")
    return silhouette_avg

def apply_spectral_clustering(user_profiles, n_clusters):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    cluster_labels = spectral.fit_predict(user_profiles)
    return cluster_labels

def apply_kmeans_clustering(user_profiles, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(user_profiles)
    return cluster_labels