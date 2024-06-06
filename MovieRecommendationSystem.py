#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


metadata = pd.read_csv("movies_metadata.csv")
ratings = pd.read_csv("ratings.csv")
credits = pd.read_csv("credits.csv")
keywords = pd.read_csv("keywords.csv")


# In[3]:


print("metadata shape:",metadata.shape)
print("metadata columns name:", metadata.columns)

print("ratings shape:",ratings.shape)
print("ratings columns name:", list(ratings.columns))

print("credits shape:",credits.shape)
print("columns name:", list(credits.columns))

print("keywords shape:",keywords.shape)
print("columns name:", list(keywords.columns))


# In[5]:


metadata = pd.read_csv("movies_metadata.csv") 
metadata = metadata.iloc[0:10000,:]

# Convert IDs to int. Required for merging on id using pandas .merge command
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe: this will look
#for candidates on the credits and keywords tables that have ids that match those
#in the metadata table, which we will use as our main data from now on.
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')
metadata.shape


# In[6]:


metadata


# 

# In[7]:


from ast import literal_eval
#Parse the stringified features into their corresponding python objects
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
  metadata[feature] = metadata[feature].apply(literal_eval)


# In[10]:


'''the crew list for a particular movie has one dictionary object per crew member. 
Each dictionary has a key called 'job' which tells us if that person was the director or not. 
With that in mind we can create a function to extract the director:'''

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
  
  
  
#Getting a list of the actors, keywords and genres
def get_list(x):
    if isinstance(x, list): #checking to see if the input is a list or not
        names = [i['name'] for i in x] #if we take a look at the data, we find that
        #the word 'name' is used as a key for the names actors, 
        #the actual keywords and the actual genres
        
        #Check if more than 3 elements exist. If yes, return only first three. 
        #If no, return entire list. Too many elements would slow down our algorithm 
        #too much, and three should be more than enough for good recommendations.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
  

  
  
'''
Now that we have written functions to clean up our data into director
names and listswith only the relevant info for cast, keywords and genres,
we can apply those functions to our data and see the results:
'''

metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

metadata[['title', 'cast', 'director', 'keywords', 'genres']].head()


# In[11]:


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x] #cleaning up spaces in the data
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
          
         
# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
  metadata[feature] = metadata[feature].apply(clean_data)

metadata.head()


# In[12]:


#This function makes use of the property of the cosine similarity funciton that
#the order and types of inputs don't matter, what matters is the similarity
#between different soups of words

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

metadata['soup'] = metadata.apply(create_soup, axis=1)
metadata[['title', 'soup', 'cast', 'director', 'keywords', 'genres']].head()


# In[13]:


#Getting the user's input for genre, actors and directors of their liking.

def get_genres():
  genres = input("What Movie Genre are you interested in (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  genres = " ".join(["".join(n.split()) for n in genres.lower().split(',')])
  return genres

def get_actors():
  actors = input("Who are some actors within the genre that you love (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  actors = " ".join(["".join(n.split()) for n in actors.lower().split(',')])
  return actors

def get_directors():
  directors = input("Who are some directors within the genre that you love (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  directors = " ".join(["".join(n.split()) for n in directors.lower().split(',')])
  return directors

def get_keywords():
  keywords = input("What are some of the keywords that describe the movie you want to watch, like elements of the plot, whether or not it is about friendship, etc? (if multiple, please separate them with a comma)? [Type 'skip' to skip this question] ")
  keywords = " ".join(["".join(n.split()) for n in keywords.lower().split(',')])
  return keywords

def get_searchTerms():
  searchTerms = [] 
  genres = get_genres()
  if genres != 'skip':
    searchTerms.append(genres)

  actors = get_actors()
  if actors != 'skip':
    searchTerms.append(actors)

  directors = get_directors()
  if directors != 'skip':
    searchTerms.append(directors)

  keywords = get_keywords()
  if keywords != 'skip':
    searchTerms.append(keywords)
  
  return searchTerms


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def make_recommendation(metadata=metadata):
  new_row = metadata.iloc[-1,:].copy() #creating a copy of the last row of the 
  #dataset, which we will use to input the user's input
  
  #grabbing the new wordsoup from the user
  searchTerms = get_searchTerms()  
  new_row.iloc[-1] = " ".join(searchTerms) #adding the input to our new row
  
  #adding the new row to the dataset
  metadata = pd.concat([metadata, pd.DataFrame([new_row])], ignore_index=True)
  
  #Vectorizing the entire matrix as described above!
  count = CountVectorizer(stop_words='english')
  count_matrix = count.fit_transform(metadata['soup'])

  #running pairwise cosine similarity 
  cosine_sim2 = cosine_similarity(count_matrix, count_matrix) #getting a similarity matrix
  
  #sorting cosine similarities by highest to lowest
  sim_scores = list(enumerate(cosine_sim2[-1,:]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  #matching the similarities to the movie titles and ids
  ranked_titles = []
  for i in range(1, 11):
    indx = sim_scores[i][0]
    ranked_titles.append([metadata['title'].iloc[indx], metadata['imdb_id'].iloc[indx]])
  
  return ranked_titles

#let's try our recommendation function now
make_recommendation()


# In[ ]:




