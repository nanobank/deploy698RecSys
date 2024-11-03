import streamlit as st
import pickle
import pandas as pd
from surprise import SVD, Dataset
import requests

# Define a function to load the data from GitHub
@st.cache_data  # Cache the data to avoid re-downloading
def load_data():
    url = "https://github.com/nanobank/deploy698RecSys/raw/main/recommendation_movie_svd.pkl"  # Replace with your GitHub file URL
    try:
        # Fetch and load the pickle data
        response = requests.get(url)
        response.raise_for_status()
        data = pickle.load(BytesIO(response.content))

        # Extract components from the loaded data
        svd_model = data['svd_model']
        movie_ratings = data['movie_ratings']
        movies = data['movies']
        return svd_model, movie_ratings, movies
    except requests.exceptions.RequestException as e:
        st.error("Error loading data: {}".format(e))
        return None, None, None

# Load the data
svd_model, movie_ratings, movies = load_data()

# Check if data is successfully loaded
if svd_model is None or movie_ratings is None or movies is None:
    st.error("Failed to load the recommendation data. Please check the file URL and format.")
else:
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
    with st.spinner('Predicting ratings...'):
        for movie_id in unrated_movies:
            prediction = svd_model.predict(user_id, movie_id)
            pred_rating.append(prediction)

    # Sort and display top recommendations
    sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
    top_rec = sorted_predictions[:num_recommendations]

    # Display recommendations in an interactive table
    st.subheader(f"Top {num_recommendations} Recommendations for User {user_id}:")
    recommendations_df = pd.DataFrame({
        'Movie Title': [movies[movies['movieId'] == prediction.iid]['title'].values[0] for prediction in top_rec],
        'Estimated Rating': [prediction.est for prediction in top_rec]
    })
    st.dataframe(recommendations_df)

    # Add a celebratory animation
    st.balloons()
