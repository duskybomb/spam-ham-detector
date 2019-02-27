import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
import string
import nltk

from preprocessing import cleanText, getPreprocessedData


def getCountVectorizer(data):
    ngram_vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer=cleanText)
    X_CountVectorizer_fit = ngram_vectorizer.fit(data['body_text'])
    return X_CountVectorizer_fit

def getTfidfVectorizer(data):
    tfidf_vectorizer = TfidfVectorizer(analyzer=cleanText)
    X_tfidfVectorizer_fit = tfidf_vectorizer.fit(data['body_text'])
    return X_tfidfVectorizer_fit

def countPunctuation(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

def countCaps(text):
    count = sum([1 for char in text if char == char.upper() ])
    return count

def makeFeatures(data):
    data = getPreprocessedData(data)
    data['body_text_tokenize'] = data['body_text_clean'].apply(lambda x: nltk.tokenize.word_tokenize(x))
    data['punctuation_percentage'] = data['body_text'].apply(lambda x: countPunctuation(x))
    data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
    data['capital_words'] = data['body_text'].apply(lambda x: countCaps(x))
    data['link_count'] = data['body_text'].apply(lambda x: len(re.findall('(((https?|ftp|smtp):\/\/)|(www.))[a-z0-9]+\.[a-z]+(\/[a-zA-Z0-9#]+\/?)*', x)))
    return data


# if __name__ == "__main__":
#     data = pd.read_csv('SMSSpamCollection.tsv', sep='\t',
#                        names=['label', 'body_text'], header=None)
#     vectorizer, X_vector = getTfidfVectorizer(data)
#     data = makeFeatures(data)
