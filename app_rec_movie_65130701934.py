import streamlit as st
import pickle

# Define a function to load the data from GitHub
@st.cache_data  # Cache the data to avoid re-downloading
def load_data():
    with open('./recommendation_movie_svd.pkl', 'rb' ) as file:
        svd_model,movie_ratings,movies = pickle.load(file)
    return svd_model, movie_ratings, movies

# Load the data
svd_model, movie_ratings, movies = load_data()
# Streamlit app title
st.title("Movie Recommender")
# Check if data is successfully loaded
