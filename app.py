import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TMDB API KEY
API_KEY = "93152e11e7c8b84aff7aead61946e809"

# load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

df = df[['title','genres']]
df = df.dropna()

df['genres'] = df['genres'].astype(str)

cv = CountVectorizer(max_features=5000, stop_words='english')
matrix = cv.fit_transform(df['genres'])

similarity = cosine_similarity(matrix)


# Function to fetch movie poster
def fetch_poster(movie_name):

    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    data = requests.get(url).json()

    if data["results"]:
        poster_path = data["results"][0]["poster_path"]
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path

    return None


# Recommendation function
def recommend(movie):

    movie_index = df[df['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:
        movie_title = df.iloc[i[0]].title
        recommended_movies.append(movie_title)
        recommended_posters.append(fetch_poster(movie_title))

    return recommended_movies, recommended_posters


# Streamlit UI
st.title("🎬 AI Movie Recommendation System")
st.write("Get movie suggestions based on content similarity using Machine Learning.")

movie_list = df['title'].values

selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):

    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])

    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])

    with col4:
        st.text(names[3])
        st.image(posters[3])

    with col5:
        st.text(names[4])
        st.image(posters[4])
