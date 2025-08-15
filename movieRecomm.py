import pandas as pd
import nltk
df=pd.read_csv('movies.csv')
df.head()
credit=pd.read_csv("credit.csv")
credit
df=pd.read_csv("movies.csv")
df=df.merge(credit,on="title")
df.head()
df['original_language'].value_counts()
df.info()
df.columns
df_selected = df[['genres', 'id', 'keywords', 'overview', 'tagline', 'title']]
df.isnull().sum()
df.dropna(inplace=True)
df.isnull()
df.duplicated().sum()
df.head()
df.iloc[0]['keywords']
import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])

    return L
df['keywords']=df['keywords'].apply(convert)
df['genres']=df['genres'].apply(convert)
df['genres'][1]
df.iloc[19].genres
def convert_cast(obj):
    L=[]
    ct=0
    for i in ast.literal_eval(obj):
        if ct!=3:
            L.append(i['name'])
            ct=ct+1
        else:
            break
    return L

df['cast']=df['cast'].apply(convert_cast)
df['cast'][0]
def fetch_director(obj):
    L=[]
    
    for i in ast.literal_eval(obj):
      if i['job']=='Director':
            L.append(i['name'])
    return L

df['crew']=df['crew'].apply(fetch_director)
df.head(20)
df['overview']=df['overview'].apply(lambda x: x.split())

df['genres']=df['genres'].apply(lambda x:[i.replace(" ","")for i in x])
df['keywords']=df['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
df['cast']=df['cast'].apply(lambda x:[i.replace(" ","")for i in x])
df['crew']=df['crew'].apply(lambda x:[i.replace(" ","")for i in x])
df.head()

df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']

df = df[['id','title','tags']]

df.head(20)
df['tags'][0]
from  sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(df['tags']).toarray()
vector.shape
vector.shape[0]
from sklearn.metrics.pairwise import cosine_similarity


similarity = cosine_similarity(vector)
print(similarity.shape)
print(df)
df.iloc[7]['title']

print(df[df['title']=='The Avengers'])
print(df[df['title']=='The Avengers'].id)
print(df[df['title']=='The Avengers'].id.values[0])
print(df[df['title']=='The Avengers'].index[0])

movie_index = df[df['title']=='The Avengers'].index[0]
print(movie_index)
print()
distance=similarity[movie_index]
print(distance)
print()
print(list(enumerate(distance)))
print()
movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
print(movie_list)
print()
for i in movie_list:
    print(df.iloc[i[0]].title)

    movie_index = df[df['title']=='The Avengers'].index[0]
distance=similarity[movie_index]

for i in movie_list:
    print(df.iloc[i[0]].title)

    def recommend(movie):
        try:
           movie_index = df[df['title'] == movie].index[0]
           distance = similarity[movie_index]
           movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
           recommendations = []
           for i in movie_list:
            print(df.iloc[i[0]].title)
        except IndexError:
          print(df, "Movie not found in the database.")

        
print(recommend('The Avengers'))






