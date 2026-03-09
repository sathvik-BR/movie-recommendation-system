import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load dataset
df = pd.read_csv("tmdb_5000_movies.csv")

# keep only important columns
df = df[['title','genres']]

# remove empty values
df = df.dropna()

# show first 5 rows
print(df.head())

df['genres'] = df['genres'].astype(str)

cv = CountVectorizer(max_features=5000, stop_words='english')

matrix = cv.fit_transform(df['genres'])

similarity = cosine_similarity(matrix)
def recommend(movie):

    movie_index = df[df['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    print("Recommended movies:")

    for i in movies_list:
        print(df.iloc[i[0]].title)

movie_name = input("Enter a movie name: ")

recommend(movie_name)