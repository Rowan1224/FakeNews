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

vocab = set()
vector = {}
with open('/media/MyDrive/Project Fake news/Models/cc.bn.300.vec', 'r', encoding="utf8") as file:
    for i, line in enumerate(file):
        l = line.split(' ')
        v = [float(i) for i in l[1:]]
        w = l[0]
        vector[w] = np.array(v)
        vocab.add(w)


print(len(vocab))
puncList = ["।", "”", "“", "’"]
stopWords = set()
for p in string.punctuation.lstrip():
    puncList.append(p)

with open('../Data/Corpus/StopWords', 'r') as f:
    for row in f:
        row = row.replace("\n", "")
        stopWords.add(row)

def tokenizer(doc):
    # remove punctuation
    tokens = []
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


def doc2MeanValue(doc):
    tokens = tokenizer(doc)
    tokentovaluelist = []
    for token in tokens:
        if token in vocab:
         tokentovaluelist.append(np.mean(vector[token]))

    return np.sum(np.array(tokentovaluelist))


df = pd.read_csv("../Data/Corpus/AllDataTarget.csv", )

featureVector = []
for row in df.iterrows():
    row = row[1]
    id = row["articleID"]
    mean = doc2MeanValue(row["news"])
    label = row["label"]
    featureVector.append((id, mean, label))

df = pd.DataFrame(featureVector, columns=["articleID", 'mean', 'label'])
print(df.shape)

df.to_csv("../Data/Corpus/word-embedding-FT-SUM-Alldata.csv", index=None, header=True)




