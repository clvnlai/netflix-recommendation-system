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

## Sample Output
```
Enter a movie or TV show name: peaky blinderz
Top Recommendations:
(Matched 'peaky blinderz' to closest title: 'Peaky Blinders')

8293                              The Fear
8334                 The Great Train Robbery
7140       Jonathan Strange & Mr Norrell
3503                       Criminal: Spain
3361                                Tunnel
8431                 The Murder Detectives
3589                          Sacred Games
5752                               Spotless
2736                       Man Like Mobeen
2606                        Extracurricular
Name: title, dtype: object
```

## Limitations & future work
- Content-based only; it doesn't use user ratings or viewing history.
- Could add cast/director features, or a web UI with Streamlit.

## Acknowledgements
Inspired by [Netflix Recommendation System using Python](https://amanxai.com/2022/07/05/netflix-recommendation-system-using-python/) by AmanXai. The base approach (TF-IDF + cosine similarity) follows that tutorial. My additions:
- Fed NLTK-cleaned genres + descriptions into the model (the original used genres only)
- Typo-tolerant title search with normalization and fuzzy matching (`difflib`)
- Fixed the queried title appearing in its own recommendations
