from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

app = Flask(__name__)

# Load and process the data as per your current code
metadata = pd.read_csv("movies_metadata.csv").iloc[0:10000, :]
ratings = pd.read_csv("ratings.csv")
credits = pd.read_csv("credits.csv")
keywords = pd.read_csv("keywords.csv")

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

# Parse stringified features into their corresponding Python objects
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)


# Define helper functions (get_director, get_list, clean_data, create_soup)
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


metadata['director'] = metadata['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    elif isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        return ''


features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

metadata['soup'] = metadata.apply(
    lambda x: ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres']),
    axis=1)


# Define recommendation function
def make_recommendation(searchTerms):
    new_row = metadata.iloc[-1, :].copy()
    new_row.iloc[-1] = " ".join(searchTerms)
    metadata_new = pd.concat([metadata, pd.DataFrame([new_row])], ignore_index=True)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata_new['soup'])

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    sim_scores = list(enumerate(cosine_sim2[-1, :]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    ranked_titles = []
    for i in range(1, 11):
        indx = sim_scores[i][0]
        ranked_titles.append([metadata['title'].iloc[indx], metadata['imdb_id'].iloc[indx]])

    return ranked_titles


# Flask Routes
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    genres = request.form.get('genres', '')
    actors = request.form.get('actors', '')
    directors = request.form.get('directors', '')
    keywords = request.form.get('keywords', '')

    searchTerms = []
    if genres:
        searchTerms.append(genres.lower().replace(" ", ""))
    if actors:
        searchTerms.append(actors.lower().replace(" ", ""))
    if directors:
        searchTerms.append(directors.lower().replace(" ", ""))
    if keywords:
        searchTerms.append(keywords.lower().replace(" ", ""))

    recommendations = make_recommendation(searchTerms)
    return render_template('results.html', recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
