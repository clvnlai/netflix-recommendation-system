import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("netflix_titles.csv")
print(data.head())

print(data.isnull().sum())

data = data[["title", "description", "type", "listed_in"]]
print(data.head())