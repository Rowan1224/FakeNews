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
import pickle
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


def word_emb():

    vocab = set()
    vector = {}
    we300 = '../Data/cc.bn.300.vec'
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

    df = pd.read_csv("../Data/Corpus/AllDataTarget.csv", )

    featureVector = []
    labels = []
    for row in df.iterrows():
        row = row[1]
        id = row["articleID"]
        mean = doc2MeanValue(row["news"])
        if mean.size == 0:
            continue
        mean = np.mean(mean, axis=0)
        # print(mean.shape)
        label = row["label"]
        r = [id, label]
        mean = (mean.tolist())
        labels.append(label)
        featureVector.append(mean)

    df = pd.DataFrame(featureVector)
    return df


def tfidf_charF(X):
    tfidf_char = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(3, 3), stop_words=stopWords,
                                 decode_error='replace', encoding='utf-8', analyzer='char')

    tfidf_char.fit(X.values.astype('U'))
    x_char = tfidf_char.transform(X.values.astype('U'))
    outfile = open("../API/tfidf_char_pkl", 'wb')
    pickle.dump(tfidf_char, outfile)
    outfile.close()
    return x_char


def tfidf_wordF(X):
    tfidf_word = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 1),
                                 stop_words=stopWords, decode_error='replace',
                                 encoding='utf-8', analyzer='word', tokenizer=tokenizer)


    x_word = tfidf_word.fit_transform(X.values.astype('U'))
    # outfile = open("tfidf_word_pkl", 'wb')
    # pickle.dump(x_word, outfile)
    # outfile.close()
    return x_word


def mp():
    dfMP = pd.read_csv("../Data/Corpus/AllDataFeature.csv")
    id = dfMP["articleID"].values
    for i in id:
        if i not in consistentID:
            dfMP = dfMP[dfMP["articleID"] != i]
    dfMP = dfMP[["nesa", "rank"]]
    dfMP = dfMP.fillna(0)
    return dfMP


df = pd.read_csv("../Data/Corpus/AllDataTarget.csv")
df = df[df["articleID"] != 27753]
consistentID = set(df["articleID"])
print(df.shape)
head = list(df)
X = df.news




# dfPOS = pd.read_csv("../Data/Corpus/AllPOSDataNLTK_N_STD.csv")
# id = dfPOS["articleID"].values
# for i in id:
#     if i not in consistentID:
#         dfPOS = dfPOS[dfPOS["articleID"] != i]
#
# dfPOS = dfPOS.fillna(0)
# dfPOS = dfPOS.drop(['articleID', 'Unnamed: 0'], axis=1)
# print(dfPOS.shape)
# X_POS = sparse.csr.csr_matrix(dfPOS.values)

# dfEmb = word_emb()
# X_Emb = sparse.csr.csr_matrix(dfEmb.values)

X_char = tfidf_charF(X)
# X_word = tfidf_wordF(X)

# dfMP = mp()
# X_MP = sparse.csr.csr_matrix(dfMP.values)

# X = sparse.hstack((X_word, X_char, X_Emb, X_MP))
X = X_char
Y = df[["label"]]

print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y.values.ravel(), test_size=0.3, random_state=109)

# Create a LR Classifier
clf = svm.SVC(kernel='linear', C=10, cache_size=7000)

# clf = LogisticRegression()

#Train the model using the training sets
clf.fit(X_train, y_train)
outfile = open("../API/model", 'wb')
pickle.dump(clf, outfile)
outfile.close()

#Predict the response for test dataset
y_pred = clf.predict(X_test)


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Precision:", metrics.precision_score(y_test, y_pred))

print("Recall:", metrics.recall_score(y_test, y_pred))

print("F1-Score:", metrics.f1_score(y_test, y_pred))

print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))

