
!pip install streamlit
import streamlit as st
import pickle

# Load data (assuming 'recommendation_movie_svd.pkl' is in the same directory)
with open('recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Streamlit app title
st.title("Movie Recommender")

# User input for user ID
user_id = st.number_input("Enter User ID:", min_value=1, value=1, step=1)

# Get recommendations
rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]

