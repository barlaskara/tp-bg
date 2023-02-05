from sklearn.feature_extraction.text import TfidfVectorizer

# List of documents
documents = ["this is the first document", "this is the second document", "and this is the third one"]

# Create the TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the documents
X = vectorizer.fit_transform(documents)

# Get the vocabulary
vocab = vectorizer.vocabulary_

# Get the term-document matrix
td_matrix = X.toarray()
print(td_matrix)