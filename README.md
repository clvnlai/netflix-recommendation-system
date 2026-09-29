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
PS C:\Users\user\Desktop\selfprojects\netflix-recommendation-system>  c:; cd 'c:\Users\user\Desktop\selfprojects\netflix-recommendation-system'; & 'C:\Users\user\AppData\Local\Microsoft\WindowsApps\python3.11.exe' 'c:\Users\user\.vscode\extensions\ms-python.debugpy-2026.6.0-win32-x64\bundled\libs\debugpy\launcher' '53216' '--' 'C:\Users\user\Desktop\selfprojects\netflix-recommendation-system\netflix_recommendations_system.py' 
[nltk_data] Downloading package stopwords to
[nltk_data]     C:\Users\user\AppData\Roaming\nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
title             0
description       0
type              0
listed_in         0
original_title    0
dtype: int64
                   title                                        description     type                                          listed_in
0   Dick Johnson Is Dead  As her father nears the end of his life, filmm...    Movie                                      Documentaries
1          Blood & Water  After crossing paths at a party, a Cape Town t...  TV Show    International TV Shows, TV Dramas, TV Mysteries
2              Ganglands  To protect his family from a powerful drug lor...  TV Show  Crime TV Shows, International TV Shows, TV Act...
3  Jailbirds New Orleans  Feuds, flirtations and toilet talk go down amo...  TV Show                             Docuseries, Reality TV
4           Kota Factory  In a city of coaching centers known to train I...  TV Show  International TV Shows, Romantic TV Shows, TV ...
826                           Bo Burnham: Inside
1509        Ariana grande: excuse me, i love you
8024                                      Single
5448               Amelia: A Tale of Two Sisters
7313    Little Lunch: The Halloween Horror Story
3234                    What the F* Is Going On?
2539                           Fire in the Blood
3903                               Someone Great
5892                                      Circle
7582                                     Nibunan
Name: title, dtype: object
Enter a movie or TV show name: peaky blinderz

Top Recommendations:
(Matched 'peaky blinderz' to closest title: 'Peaky Blinders')

8293                         The Fear
8334          The Great Train Robbery
7140    Jonathan Strange & Mr Norrell
3503                  Criminal: Spain
3361                           Tunnel
8431            The Murder Detectives
3589                     Sacred Games
5752                         Spotless
2736                  Man Like Mobeen
2606                  Extracurricular
Name: title, dtype: object
PS C:\Users\user\Desktop\selfprojects\netflix-recommendation-system> 

## Limitations & future work
- Content-based only; it doesn't use user ratings or viewing history.
- Could add cast/director features, or a web UI with Streamlit.
