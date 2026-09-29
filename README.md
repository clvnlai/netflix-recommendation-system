# Netflix Recommendation System

A content-based recommender that suggests Netflix movies and TV shows similar to a title you enter, using NLP preprocessing (NLTK) and TF-IDF with cosine similarity.

## How it works
1. **Text cleaning (NLTK):** lowercases text, removes URLs, punctuation and numbers, drops stopwords, and applies Snowball stemming.
2. **Feature extraction:** genres and descriptions are combined and converted into TF-IDF vectors.
3. **Similarity:** cosine similarity between vectors finds the closest titles.
4. **Title matching:** input is normalized (case, spaces, punctuation) and falls back to fuzzy matching (`difflib`) to handle typos, e.g. `peaky blinderz` -> `Peaky Blinders`.

## Tech stack
Python, pandas, NLTK, scikit-learn, difflib

## Dataset
[Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) (Kaggle), about 8,800 titles.
Download `netflix_titles.csv` and place it in the project folder.

## Setup
```bash
pip install pandas nltk scikit-learn
python netflix_recommendations_system.py
```

## Example
```
Enter a movie or TV show name: peaky blinderz
(Matched 'peaky blinderz' to closest title: 'Peaky Blinders')

Top Recommendations:
1  ...
2  ...

```


## Limitations & future work
- Content-based only; it doesn't use user ratings or viewing history.
- Could add cast/director features, or a web UI with Streamlit.
