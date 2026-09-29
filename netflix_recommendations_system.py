import numpy as np
import pandas as pd
import nltk
import re
import string
import difflib
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words("english"))

data = pd.read_csv("netflix_titles.csv")
data = data[["title", "description", "type", "listed_in"]].dropna().reset_index(drop=True)

data["original_title"] = data["title"]

print(data.isnull().sum())

data = data[["title", "description", "type", "listed_in"]]
print(data.head())

data = data.dropna()

def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)                    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)     
    text = re.sub(r'<.*?>+', '', text)                     
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)                  
    text = [word for word in text.split() if word not in stopword]   
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split()]             
    text=" ".join(text)
    return text

data["clean_title"] = data["title"].apply(clean)

print(data.title.sample(10))

# Building tfidf inputs from genres and description
data["combined"] = (data["listed_in"] + " " + data["description"]).apply(clean)
feature = data["combined"].tolist()
tfidf = text.TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)

def normalize(s):
    return re.sub(r'[\W_]+', '', str(s).lower())

normalized_to_index = {}
for i, t in enumerate(data["title"]):
    normalized_to_index.setdefault(normalize(t), i)

normalized_titles = list(normalized_to_index.keys())

def find_title_index(query):
    """Returns (index, was_corrected) or (None, False)."""
    key = normalize(query)

    if key in normalized_to_index:
        return normalized_to_index[key], False

# Function to recommend movies and shows on Netflix
def netFlix_recommendation(title, similarity=similarity):
    idx, was_corrected = find_title_index(title)

    if idx is None:
        return f"Sorry, '{title}' was not found in the dataset. Please check the spelling!"
    
    if was_corrected:
        matched_title = data['title'].iloc[idx]
        print(f"(Matched '{title}' to closest title: '{matched_title}')\n")

    index = indices_lower[cleaned_title]
    similarity_scores = list(enumerate(similarity[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    similarity_scores = [s for s in similarity_scores if s[0] != index][:10]
    
    movieindices = [i[0] for i in similarity_scores]
    return data['title'].iloc[movieindices]

user_movie = input("Enter a movie or TV show name: ")
print("\nTop Recommendations:")
print(netFlix_recommendation(user_movie))