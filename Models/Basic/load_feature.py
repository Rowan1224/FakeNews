import os
import pandas as pd
import re
import numpy as np
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import string
import pickle
import sys
sys.path.insert(1, '../Helper/')
import helper, config


def getStopWords():
    stopWords = []
    with open(config.STOP_WORD, 'r', encoding="utf8") as f:
        for row in f:  # iterate over the rows in the file
            row = row.replace("\n", "")
            stopWords.append(row)

    return stopWords

def tokenizer(doc):
    puncList = ["।", "”", "“", "’"]
    for p in string.punctuation.lstrip():
        puncList.append(p)
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


def word_emb(size,X):

    vocab = set()
    vector = {}

    we300 = config.EMBEDDING_300
    we100 = config.EMBEDDING_100

    if size == 100:
        w = we100
    else:
        w = we300

    with open(w, 'r', encoding="utf8") as file:
        for _, line in enumerate(file):
            l = line.split(' ')
            v = [float(i) for i in l[1:]]
            w = l[0]
            vector[w] = np.array(v)
            vocab.add(w)

    print("Vocab Size =>%s" %(len(vocab)))

    def doc2MeanValue(doc):
        tokens = tokenizer(doc)
        tokentovaluelist = [vector[token] for token in tokens if token in vocab]
        return np.array(tokentovaluelist)

    df =  X

    featureVector = []
    labels = []
    for row in df.iterrows():
        row = row[1]
        # id = row["articleID"]
        mean = doc2MeanValue(row["content"])
        if mean.size == 0:
            mean = [0] * size
            featureVector.append(mean)
            continue
        mean = np.mean(mean, axis=0)
        label = row["label"]
        mean = (mean.tolist())
        labels.append(label)
        featureVector.append(mean)

    df = pd.DataFrame(featureVector)
    df = df.fillna(0)
    return sparse.csr.csr_matrix(df.values)


def tfidf_charF(X, a, b, save_model=False):


    X = X.content
    tfidf_char = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(a, b), stop_words=getStopWords(),
                                 decode_error='replace', encoding='utf-8', analyzer='char')

    tfidf_char.fit(X.values.astype('U'))
    if save_model:
        name = "tfidf_char.pkl"
        path = config.API+name
        outfile = open(path, 'wb')
        pickle.dump(tfidf_char, outfile)
        outfile.close()
    x_char = tfidf_char.transform(X.values.astype('U'))
    return x_char


def tfidf_wordF(X, a, b):

    X = X.content
    tfidf_word = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(a, b),
                                 stop_words=getStopWords(), decode_error='replace',
                                 encoding='utf-8', analyzer='word', tokenizer=tokenizer)
    
    tfidf_word.fit(X.values.astype('U'))

    if save_model:
        name = "tfidf_word.pkl"
        path = config.API+name
        outfile = open(path, 'wb')
        pickle.dump(tfidf_char, outfile)
        outfile.close()
    
    x_word = tfidf_word.transform(X.values.astype('U'))
    return x_word


def mp(X):

    def count_punc(content):
        char_list = list(content)
        count = 0
        for c in char_list:
            if c == '!':
                count += 1
        return count

    def load_rank():
        rank = {}
        mx = 0
        na = []
        paths = os.listdir(config.ALEXA_RANK)
        for path in paths:
            d = pd.read_csv(config.ALEXA_RANK+"/"+path)
            # d = d.drop(['Unnamed: 0'], axis=1)
            d = d.fillna(0)
            # print(d[d['rank'].isnull()])
            for row in d.iterrows():
                row = row[1]
                if row['rank'] == 0:
                    na.append(row['domain'])
                else:
                    rank[row['domain']] = int(row['rank'])
                    mx = max(int(row['rank']), mx)
        for n in na:
            rank[n] = mx
        return rank

    ranks = load_rank()
    df = X
    featureVector = []
    for row in df.iterrows():
        row = row[1]
        feature = []
        feature.append(count_punc(row['content']))
        r = ranks[row['domain']]
        feature.append(r)
        featureVector.append(feature)

    dfMP = pd.DataFrame(featureVector)
    normalized_df = (dfMP - dfMP.mean()) / dfMP.std()
    dfMP = normalized_df.fillna(0)
    return sparse.csr.csr_matrix(dfMP.values)


def pos():
    path_pos = config.POS_PATH
    dfPOS = pd.read_csv(path_pos)
    dfPOS = dfPOS.fillna(0)
    dfPOS = dfPOS.drop(['articleID', 'Unnamed: 0'], axis=1)
    X_POS = sparse.csr.csr_matrix(dfPOS.values)
    return X_POS





