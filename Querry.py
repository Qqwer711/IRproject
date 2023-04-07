from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

with open('article_titles_stemmed.txt', 'r') as f:
    corpus = [f.read()]

vectorizer = TfidfVectorizer()


tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()


query = input("Enter your query: ")


query_vector = vectorizer.transform([query])


similarity_scores = cosine_similarity(query_vector, tfidf_matrix)   


for i, score in enumerate(similarity_scores[0]):
    print(f"in Document {i+1}: {score:.5f}")