!pip install scikit-surprise
import streamlit as st
import pickle
import scikit-learn==1.2.2
import scikit-surprise
import pandas
import numpy==1.26.4


# Load data only once and cache it
@st.cache_data  # This decorator caches the data for faster loading
def load_data():
    with open('recommendation_movie_svd.pkl', 'rb') as file:
        svd_model, movie_ratings, movies = pickle.load(file)
    return svd_model, movie_ratings, movies

# Load the data
svd_model, movie_ratings, movies = load_data()

# Streamlit app title
st.title("Movie Recommender")

# User input for user ID
user_id = st.number_input("Enter User ID:", min_value=1, value=1, step=1)

# Get recommendations
rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
top_rec = sorted_predictions[:10]  

# Display recommendations
st.subheader("Top 10 Recommendations:")
for prediction in top_rec:
    st.write(f"Movie ID: {prediction.iid}, Estimated Rating: {prediction.est}")
