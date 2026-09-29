# This is for TfidVectorizer 

# note to self: this is TfidVectorizer, when e.g. comparing data (like, text) with  cosine similarity  
```python
feature = data["listed_in"].tolist()
# this is setting up the rules of transformation
tfidf = text.TfidfVectorizer(stop_words="english")
# this is to transform existing dataset
tfidf_matrix = tfidf.fit_transform(feature)
# calculate cosine similarity
similarity = cosine_similarity(tfidf_matrix)
```
# mapping movie "title" in this case to dataset indices for fast recommendation

```python
indices = pd.Series(data.index,index=data['title']).drop_duplicates()   
```

# fuzzy matching (handles spaces or typo)