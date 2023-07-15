from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

# List of documents
documents = ["w1 w2 w3 w6", "w1 w2 w3 w5 w6", "w4 w5 w6 w7"]

# Create the TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the documents
X = vectorizer.fit_transform(documents)

# Get the vocabulary
vocab = vectorizer.vocabulary_
print(vocab)
# Get the term-document matrix
td_matrix = np.matrix(X.toarray())
td_matrix = td_matrix.round(2)

print(td_matrix)
A = {1, 2, 3}
B = {1, 2, 3, 4, 5}

print(A.issubset(B))  # Output: True

C = {1, 2, 4}
print(C.issubset(B))  # Output: True

D = {1, 2, 3, 4, 5, 6}
print(D.issubset(B))  # Output: False


def set_coverage(S, Sets):
    return len([X for X in Sets if X.issubset(S)])


S = {1, 2, 3, 4, 5}
Sets = [{1, 2}, {3, 4}, {5}, {1, 3, 5}]

print(set_coverage(S, Sets))  # Output: 4
print(set_coverage({1, 2}, Sets))  # Output: 1

