#!pip install scikit-surprise
import streamlit as st
import pickle
import scikit-learn==1.2.2
#import scikit-surprise as surprise
import scikit-surprise
import pandas
import numpy==1.26.4


import streamlit as st
import pickle
import pandas as pd
from scikit-surprise import SVD
#from surprise import SVD
from scikit-surprise import Dataset
#from surprise import Dataset
from scikit-surprise.model_selection import cross_validate
#from surprise.model_selection import cross_validate

# Load data from URL (replace with your actual URL)
@st.cache_data  # Cache the data for faster loading
def load_data():
    url = "https://github.com/nanobank/deploy698RecSys/blob/main/recommendation_movie_svd.pkl"  # Replace with your file URL
    data = pd.read_pickle(url)
    svd_model = data['svd_model']
    movie_ratings = data['movie_ratings']
    movies = data['movies']
    return svd_model, movie_ratings, movies

# Load the data
svd_model, movie_ratings, movies = load_data()

# Streamlit app title
st.title("Movie Recommender")

# User input for user ID with a sidebar
user_id = st.sidebar.number_input("Enter User ID:", min_value=1, value=1, step=1)

# Number of recommendations to display
num_recommendations = st.sidebar.slider("Number of Recommendations:", min_value=1, max_value=20, value=10, step=1)

# Get recommendations
rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']

# Predict ratings with a progress bar
pred_rating = []
with st.spinner('Predicting ratings...'):  # Display a spinner while predicting
    for movie_id in unrated_movies:
        pred_rating.append(svd_model.predict(user_id, movie_id))

sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
top_rec = sorted_predictions[:num_recommendations]

# Display recommendations in an interactive table
st.subheader(f"Top {num_recommendations} Recommendations for User {user_id}:")
recommendations_df = pd.DataFrame({
    'Movie Title': [movies[movies['movieId'] == prediction.iid]['title'].values[0] for prediction in top_rec],
    'Estimated Rating': [prediction.est for prediction in top_rec]
})
st.dataframe(recommendations_df)  # Display recommendations in a table

# Add a celebratory animation
st.balloons()
