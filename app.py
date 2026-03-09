import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

df = df[['title','genres']]
df = df.dropna()

df['genres'] = df['genres'].astype(str)

cv = CountVectorizer(max_features=5000, stop_words='english')
matrix = cv.fit_transform(df['genres'])

similarity = cosine_similarity(matrix)

def recommend(movie):

    movie_index = df[df['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    recommended = []

    for i in movies_list:
        recommended.append(df.iloc[i[0]].title)

    return recommended


st.title(" Movie Recommendation System")

movie_list = df['title'].values

selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.write("Recommended Movies:")

    for movie in recommendations:
        st.write(movie)