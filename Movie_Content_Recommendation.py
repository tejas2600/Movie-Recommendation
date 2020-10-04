import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Building%20a%20Movie%20Recommendation%20Engine/movie_dataset.csv")
titles=df.loc[:,['title']]

features = ['title','keywords','cast','genres','director']

def combine_features(row):
    return row['title']+row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]

for feature in features:
    df[feature] = df[feature].fillna('')

df["combined_features"] = df.apply(combine_features,axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

movie_user_likes = "Titanic"


ispresent = False
for title in titles:
    if(movie_user_likes==title):
        ispresent =True
        break

if(ispresent==False):
    print("Movie not found")

else:
    movie_index = get_index_from_title(movie_user_likes)
    similar_movies =  list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]


    i=0
    print("Top 10 similar movies to "+movie_user_likes+" are:")
    for element in sorted_similar_movies:
        print(get_title_from_index(element[0]))
        i=i+1
        if i>=10:
            break

