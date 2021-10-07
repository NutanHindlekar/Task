#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import sklearn.metrics as metrics
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pickle


# In[9]:


app = FastAPI()
@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return '''
        <form method="post">
        <input maxlength="28" name="text" type="text" value="Text Emotion to be tested" />
        <input type="submit" />'''


# In[ ]:


d1 = pd.read_csv('channel_data_19_jul.csv')
d2 = pd.read_csv('channel_users_19_jul.csv')
d3 = d1.append(d2)
d3.drop('Unnamed: 0', axis=1 ,inplace=True)
print('Before cleaning', d3.shape)

d3.drop_duplicates(subset='video_id',inplace=True,keep="first")
d3.drop_duplicates(subset='title',inplace=True,keep="first")
d3.replace('None', np.nan, inplace = True)
d3.dropna(subset=['Tags'], inplace=True)
d3.reset_index(inplace=True, drop=True)
print('after cleaning', d3.shape)

w_token = nltk.tokenize.WhitespaceTokenizer()
lemmatisation = nltk.stem.WordNetLemmatizer()
default_stopwords = stopwords.words('english') # or any other list of your choice
exclude = set(string.punctuation) 

def clean_text(text, ):

    def _remove_punct(text):
        return ''.join(t for t in text if t not in exclude)

    def lam_txt(text):
        out_text = [lemmatisation.lemmatize(word)for word in w_token.tokenize(text)]
        return ' '.join(out_text)
    
    text = text.astype('str').apply(lambda x: re.sub(r"http\S+", "", x))
    text=text.astype('str').apply(lambda x: x.strip())
    text = text.apply(lambda x: x.lower())# lowercase
    text = text.astype('str').apply(lambda x: lam_txt(x)) # stemming
    text =text.apply(lambda x: _remove_punct(x)) # remove punctuation and symbols
    
    return text


d3['clean_Tags']=clean_text(d3['Tags'])
d3['clean_Description']=clean_text(d3['Description'])
d3['title']=d3['title'].astype('str').apply(lambda x: x.lower())
d3['title']=d3['title'].str.strip()


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(d3['clean_Tags'])
indices = pd.Series(d3.index, index=d3['title'])

Batch_size = 10000
cosine_sim = []
leng1 = count_matrix.shape[0]
i=0
j=0
while leng1 !=0:
  if leng1 <= Batch_size:
    j = leng1
    print(j)
  else:
    j = j+Batch_size
    
  leng1 = leng1-j
  cosine_sim1 = linear_kernel(count_matrix[i:i+j], count_matrix)
  print("{} completed".format(i+j))
  i = i+j
  cosine_sim.append(cosine_sim1)
  j = 0
    
    
cosine_similarity_= np.vstack(cosine_sim)

# Function that takes in video title as input and outputs most similar video
def get_recommendations(title):
    Title= title.lower()
    idx = indices[Title]                                              # Get the index of the video that matches the title
    sim_scores = list(enumerate(cosine_similarity_[idx]))                     # Get the pairwsie similarity scores of all videos with that video
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Sort the video based on the similarity scores
    sim_scores = sim_scores[1:11]                                     # Get the scores of the 10 most similar Videos
    video_indices = [i[0] for i in sim_scores]                        # Get the movie indices
    return d3['title'].iloc[video_indices]                            # Return the top 10 most similar Videos


# In[ ]:


@app.post('/predict')
def predict(text:str = Form(...)):
    clean_text = clean_text(text) #clean, and preprocess the text through pipeline
    predictions = get_recommendations(clean_text) #predict the text
    
    return { predictions
    }

