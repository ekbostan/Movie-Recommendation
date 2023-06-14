import pandas as pd
from preprocessing import clean_text, json_to_str
from movie_similarity import calculate_movie_similarity, get_top_related_movies

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Preprocess data
movies['overview'] = movies['overview'].fillna("")
movies['overview'] = movies['overview'].astype(str)
movies['overview'] = movies['overview'].apply(clean_text)
movies['keywords'] = movies['keywords'].apply(json_to_str)
movies['compound_overview'] = movies['keywords'] + " " + movies['overview']

# Calculate movie similarity
result_matrix = calculate_movie_similarity(movies)

# Get top related movies for a given input
def get_top_related_movies(movie, similarity_matrix, movies):
    movie_index = movies[movies["title"].str.lower() == movie.lower()].index
    if len(movie_index) == 0:
        print("Movie not found. Please try again.")
        return

    movie_index = movie_index[0]
    top_related_movies = similarity_matrix.loc[movie_index, "Similar Movies"]
    return top_related_movies

# Prompt user for input
movie_input = input("Enter a movie: ")
related_movies = get_top_related_movies(movie_input, result_matrix, movies)

# Display top related movies
print("Top related movies:")
print(related_movies)
