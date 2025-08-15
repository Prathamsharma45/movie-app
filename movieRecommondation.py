#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:





# In[ ]:





# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('movies.csv')



# In[4]:


credit=pd.read_csv("credit.csv")



# In[5]:


df=pd.read_csv("movies.csv")


# In[ ]:


df=df.merge(credit,on="title")


# In[7]:





# In[8]:


df['original_language'].value_counts()


# In[9]:





# In[10]:





# In[11]:


df_selected = df[['genres', 'id', 'keywords', 'overview', 'tagline', 'title']]


# In[12]:


df.isnull().sum()


# In[ ]:





# In[13]:


df.dropna(inplace=True)
df.isnull()


# In[14]:


df.duplicated().sum()


# In[15]:





# In[16]:


df.iloc[0]['keywords']


# In[ ]:






# import ast

# In[17]:


import ast


# In[18]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])

    return L


# In[19]:


df['keywords']=df['keywords'].apply(convert)



# In[20]:


df['genres']=df['genres'].apply(convert)


# In[21]:


df['genres'][1]


# In[22]:


df.iloc[19].genres


# In[ ]:





# In[23]:


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


# In[24]:


df['cast']=df['cast'].apply(convert_cast)


# In[25]:


df['cast'][0]


# In[ ]:





# In[26]:


def fetch_director(obj):
    L=[]

    for i in ast.literal_eval(obj):
      if i['job']=='Director':
            L.append(i['name'])
    return L


# In[ ]:





# In[27]:


df['crew']=df['crew'].apply(fetch_director)


# In[28]:





# In[29]:


df['overview']=df['overview'].apply(lambda x: x.split())


# In[30]:


df['genres']=df['genres'].apply(lambda x:[i.replace(" ","")for i in x])
df['keywords']=df['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
df['cast']=df['cast'].apply(lambda x:[i.replace(" ","")for i in x])
df['crew']=df['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[31]:





# In[32]:


df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']


# In[33]:


df = df[['id','title','tags']]


# In[34]:



# In[35]:


df['tags'][0]


# In[ ]:





# In[36]:


df['tags']=df['tags'].apply(lambda x: " ".join(x))


# In[37]:


df['tags'] = df['tags'].apply(lambda x: x.lower())


# In[38]:


import nltk


# In[39]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[40]:


ps.stem('walked')


# In[41]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[42]:


df['tags'] = df['tags'].apply(stem)


# In[43]:


df['tags'][0]


# In[44]:


from  sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[65]:


vector = cv.fit_transform(df['tags']).toarray()


# In[46]:


vector.shape


# In[47]:


vector.shape[0]


# In[48]:


from sklearn.metrics.pairwise import cosine_similarity


# In[49]:


similarity = cosine_similarity(vector)
(similarity.shape)


# In[50]:





# In[51]:


df.iloc[7]['title']


# In[52]:


(df[df['title']=='The Avengers'])


# In[53]:


(df[df['title']=='The Avengers'].id)


# In[54]:


(df[df['title']=='The Avengers'].id.values[0])


# In[55]:


(df[df['title']=='The Avengers'].index[0])


# In[56]:


movie_index = df[df['title']=='The Avengers'].index[0]
movie_index

distance=similarity[movie_index]
(list(enumerate(distance)))
movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
movie_list

for i in movie_list:
    print(df.iloc[i[0]].title)


# In[57]:


movie_index = df[df['title']=='The Avengers'].index[0]
distance=similarity[movie_index]

for i in movie_list:
    print(df.iloc[i[0]].title)


# In[58]:


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


# In[59]:


print(recommend('The Avengers'))


# In[60]:


recommend('Avatar')


# In[64]:


recommend('Spider-Man 3')


# In[63]:

import streamlit as st

st.title("Movie Recommendation App")
st.text_input("Enter a movie title to get recommendations:")
st.button("Recommend", on_click=lambda: recommend(st.session_state.input))


# ...your code...

