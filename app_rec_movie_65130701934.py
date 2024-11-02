import streamlit as st
import pickle
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load data from URL (replace with your actual URL)
@st.cache_data  # Cache the data for faster loading
def load_data(url):
    try:
        data = pd.read_pickle(url)
        svd_model = data['svd_model']
        movie_ratings = data['movie_ratings']
        movies = data['movies']
        return svd_model, movie_ratings, movies
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return None, None, None

# Streamlit app title
st.title("Movie Recommender")

# URL input for the data file
url = st.sidebar.text_input("Enter Data URL:", "https://github.com/nanobank/deploy698RecSys/blob/main/recommendation_movie_svd.pkl")

# Load the data with error handling
svd_model, movie_ratings, movies = load_data(url)

if svd_model is not None and movie_ratings is not None and movies is not None:
    # User input for user ID with a sidebar
    user_id = st.sidebar.number_input("Enter User ID:", min_value=1, value=1, step=1)

    # Check if the entered user_id exists
    if user_id not in movie_ratings['userId'].unique():
        st.error(f"User ID {user_id} not found in the dataset.")
    else:
        # Number of recommendations to display
        num_recommendations = st.sidebar.slider("Number of Recommendations:", min_value=1, max_value=20, value=10, step=1)

        # Get recommendations
        rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
        unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']

        # Predict ratings with a progress bar
        pred_rating = []
        with st.spinner('Predicting ratings...'):  # Display a spinner while predicting
            for i, movie_id in enumerate(unrated_movies):
                pred_rating.append(svd_model.predict(user_id, movie_id))
                # Update progress bar
                progress = (i + 1) / len(unrated_movies)
                st.progress(progress)

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
