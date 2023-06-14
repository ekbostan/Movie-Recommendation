import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_movie_similarity(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["compound_overview"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    movie_similarity = pd.DataFrame(columns=["Movie A", "Similar Movies", "Similarity Scores"])

    for i, movie in enumerate(movies["title"]):
        similar_movie_indices = cosine_sim[i].argsort()[::-1]
        similar_movie_indices = similar_movie_indices[similar_movie_indices != i]
        similar_movies = [movies.loc[index, "title"] for index in similar_movie_indices[:10]]
        movie_a = movies.loc[i, "title"]
        movie_similarity = movie_similarity.append({"Movie A": movie_a, "Similar Movies": similar_movies}, ignore_index=True)

    return movie_similarity
