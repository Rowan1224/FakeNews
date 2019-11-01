import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.base import TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import re, string, collections, os, random


def vec_to_vocab(path):
    vocab = set()
    vector = {}
    with open(path, 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            l = line.split(' ')
            v = [float(i) for i in l[1:]]
            w = l[0]
            vector[w] = np.array(v)
            vocab.add(w)
            # print(len(vocab))
    return vocab



# with open('../Data/Corpus/StopWords', 'r') as f:
#     for row in f:
#         row = row.replace("\n", "")
#         stopWords.add(row)

def tokenizer(doc):
    # remove punctuation
    tokens = []
    puncList = ["।", "”", "“", "’"]
    for p in string.punctuation.lstrip():
        puncList.append(p)

    def cleanword(word):
        for p in puncList:
            word = word.replace(p, "")
        word = re.sub(r'[\u09E6-\u09EF]', "", word, re.DEBUG)  # replace digits

        return word

    for word in doc.split(" "):
        word = cleanword(word)
        if word != "":
            tokens.append(word)

    return tokens


def doc_to_vocab(doc):
    tokens = tokenizer(doc)

    for token in tokens:
        news_vocab.add(token)

    return news_vocab


df = pd.read_csv("../Data/Corpus/AllDataTarget.csv", )
news_vocab = set()
featureVector = []
for row in df.iterrows():
    row = row[1]
    mean = doc_to_vocab(row["news"])


vocab = vec_to_vocab('../Data/word2vec_model.txt')

common = vocab.intersection(news_vocab)
print("News Vocab Size: ==>", len(news_vocab))
print("Word Vocab Size: ==>", len(vocab))
print("Common Vocab Size: ==>", len(common))
print("Present in word vocab but not in news Vocab Size: ==>", len(vocab.difference(news_vocab)))
print("Present in news Vocab but not in word vocab Size: ==>", len(news_vocab.difference(vocab)))
