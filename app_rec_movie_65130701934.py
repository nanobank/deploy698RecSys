#%%writefile streamlit_app.py
import streamlit as st
import pickle

# Load recommendation data
with open('recommendation_data.pkl', 'rb') as file:
    user_similarity_df, user_movie_ratings = pickle.load(file)

# Function to get movie recommendations (from myfunction_65130701934.py)
def get_movie_recommendations(user_id, user_similarity_df, user_movie_ratings, n_recommendations=10):
    # ... (your existing function code) ...

# Streamlit app
st.title("Movie Recommender System")

user_id = st.number_input("Enter User ID:", min_value=1, max_value=user_similarity_df.shape[0], value=1, step=1)

if st.button("Get Recommendations"):
    recommendations = get_movie_recommendations(user_id, user_similarity_df, user_movie_ratings)
    st.write(f"Top 10 movie recommendations for User {user_id}:")
    for movie_title in recommendations:
        st.write("          " + movie_title)
