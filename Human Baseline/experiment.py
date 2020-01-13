import os
import pandas as pd
import re
import numpy as np
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from scipy import sparse, hstack, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import string

puncList = ["।", "”", "“", "’"]
for p in string.punctuation.lstrip():
    puncList.append(p)

stopWords = []
with open('../Data/Corpus/StopWords', 'r') as f:
    for row in f:  # iterate over the rows in the file
        row = row.replace("\n", "")
        stopWords.append(row)


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


def word_emb(train, test):
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

    def doc2MeanValue(doc):
        tokens = tokenizer(doc)
        tokentovaluelist = []
        for token in tokens:
            if token in vocab:
                tokentovaluelist.append((vector[token]))

        return np.array(tokentovaluelist)



    train_vector = []

    for row in train.iterrows():
        row = row[1]

        mean = doc2MeanValue(row["news"])
        if mean.size == 0:
            continue
        mean = np.mean(mean, axis=0)

        mean = (mean.tolist())
        train_vector.append(mean)

    test_vector = []
    for row in test.iterrows():
        row = row[1]

        mean = doc2MeanValue(row["news"])
        if mean.size == 0:
            continue
        mean = np.mean(mean, axis=0)

        mean = (mean.tolist())
        test_vector.append(mean)

    return pd.DataFrame(train_vector), pd.DataFrame(test_vector)


def tfidf_charF(train, test):
    tfidf_char = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(3, 3), stop_words=stopWords,
                                 decode_error='replace', encoding='utf-8', analyzer='char')

    tfidf_char.fit(train.values.astype('U'))
    x_char_train = tfidf_char.transform(train.values.astype('U'))
    x_char_test = tfidf_char.transform(test.values.astype('U'))

    return x_char_train, x_char_test


def tfidf_wordF(train, test):
    tfidf_word = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 1),
                                 stop_words=stopWords, decode_error='replace',
                                 encoding='utf-8', analyzer='word', tokenizer=tokenizer)

    tfidf_word.fit(train.values.astype('U'))
    x_word_train = tfidf_word.transform(train.values.astype('U'))
    x_word_test = tfidf_word.transform(test.values.astype('U'))
    return x_word_train, x_word_test


def data_to_features():
    df = pd.read_csv("../Data/Corpus/AllDataTarget.csv")
    df = df[df["articleID"] != 27753]

    tdf = pd.read_csv("HB.csv")
    df = pd.merge(df[['news', 'label']], tdf[['news', 'label']], how='left', on=["news", "label"], indicator= True)[lambda x: x._merge == 'left_only'].drop('_merge', 1)

    train = df.news
    test = tdf.news
    # df_train, df_test = word_emb(df, tdf)

    # x_emb_train = sparse.csr.csr_matrix(df_train.values)
    # x_emb_test = sparse.csr.csr_matrix(df_test.values)

    x_char_train, x_char_test = tfidf_charF(train, test)
    # x_word_train, x_word_test = tfidf_wordF(train, test)

    # x_train = sparse.hstack((x_word_train, x_char_train, x_emb_train))
    x_train = x_char_train
    y_train = df[["label"]]

    # x_test = sparse.hstack((x_word_test, x_char_test, x_emb_test))
    x_test = x_char_test
    y_test = tdf[["label"]]

    return x_train, x_test, y_train, y_test


X_train, X_test, y_train, y_test = data_to_features()


clf = svm.SVC(kernel='linear', C=10, cache_size=7000)

# clf = LogisticRegression()


clf.fit(X_train, y_train.values.ravel())

y_pred = clf.predict(X_test)

y_test = y_test.values.ravel()
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

print("Recall:", metrics.recall_score(y_test, y_pred))

print("F1-Score:", metrics.f1_score(y_test, y_pred))

print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))

