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
from sklearn import svm
# from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import re, string, collections, os, random


def classifier(X, Y):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109)

    # Create a LR Classifier
    clf = svm.SVC(kernel='linear', C=10, cache_size=7000)

    # clf = LogisticRegression()

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))

    print("F1-Score:", metrics.f1_score(y_test, y_pred))

    print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))

    print(metrics.classification_report(y_test, y_pred))


vocab = set()
vector = {}
we300 = '/media/MyDrive/Project Fake news/Models/cc.bn.300.vec'
we100 = '../Data/word2vec_model.txt'

with open(we300, 'r', encoding="utf8") as file:
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
            tokentovaluelist.append((vector[token]))
            # tokentovaluelist.append((vector[token]))

    return np.array(tokentovaluelist)


df = pd.read_csv("../Data/Corpus/AllDataTarget.csv", )

featureVector = []
labels = []
for row in df.iterrows():
    row = row[1]
    id = row["articleID"]
    mean = doc2MeanValue(row["news"])
    if mean.size == 0:
        print(id)
        continue
    mean = np.mean(mean, axis=0)
    # print(mean.shape)
    label = row["label"]
    r = [id, label]
    mean = (mean.tolist())
    labels.append(label)
    featureVector.append(mean)


df = pd.DataFrame(featureVector)
print(df.shape)
classifier(df, labels)
