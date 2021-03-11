import os
import pandas as pd
import re
import numpy as np
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from scipy import sparse, hstack, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import string
import pickle
import load_feature
import argparse

import sys
sys.path.insert(1, '../Helper/')
import helper, config


def load_data():
    data_df = pd.read_csv(config.DATA_PATH)
    return data_df



def mp(args):

    model(load_feature.mp(X),Y,"MP")


def pos(args):

    model(load_feature.pos(),Y,"POS")



#Unigram
def unigram(args):

    model(load_feature.tfidf_wordF(X, 1, 1),Y,"Unigram")


#Bigram
def bigram(args):

    model(load_feature.tfidf_wordF(X, 2, 2),Y,"Bigram")

#Trigram
def trigram(args):

    model(load_feature.tfidf_wordF(X, 3, 3),Y,"Trigram")


#U+B+T
def u_b_t(args):

    model(load_feature.tfidf_wordF(X, 1, 3),Y,"U+B+T")


#C3
def char_3(args):

    model(load_feature.tfidf_charF(X, 3, 3,True),Y,"C3-gram")

def char_4(args):

    model(load_feature.tfidf_charF(X, 4, 4),Y,"C4-gram")

def char_5(args):

    model(load_feature.tfidf_charF(X, 5, 5),Y,"C5-gram")

def char_3_4_5(args):

    model(load_feature.tfidf_charF(X, 3, 5),Y,"C3+C4+C5")


#Linguistic
def lexical(args):

    X_char = load_feature.tfidf_charF(X, 3, 5)
    X_word = load_feature.tfidf_wordF(X, 1, 3)
    model(sparse.hstack((X_word, X_char)),Y,"Lexical")


#Word Embedding Fasttext
def word_300(args):
    model(load_feature.word_emb(300,X),Y,"Emb_F")


#Word Embedding News
def word_100(args):

    model(load_feature.word_emb(100,X),Y,"Emb_N")

#L+POS
def L_POS(args):

    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos())),Y,"L+POS")


#L+POS+Emb(F)
def L_POS_Emb_F(args):
    

    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(), load_feature.word_emb(300,X))),Y,"L+POS+Emb(F)")

#L+POS+Emb(N)
def L_POS_Emb_N(args):
    
    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(), load_feature.word_emb(100,X))),Y,"L+POS+Emb(N)")


#L+POS+E(F)+MP
def L_POS_Emb_F_MP(args):
    
    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(), load_feature.word_emb(300,X), load_feature.mp(X))),Y,"L+POS+E(F)+MP")


#L+POS+E(N)+MP
def L_POS_Emb_N_MP(args):

    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(), load_feature.word_emb(100,X), load_feature.mp(X))),Y,"L+POS+E(N)+MP")


#Allfeatures
def allfeatures(args):

    model(sparse.hstack((load_feature.tfidf_charF(X, 3, 5), load_feature.tfidf_wordF(X, 1, 3), load_feature.pos(), load_feature.word_emb(300,X), load_feature.word_emb(100,X), load_feature.mp(X))),Y,"Allfeatures")



def model(X,Y,exp):

    X_train, X_test, y_train, y_test = train_test_split(X, Y.values.ravel(), test_size=0.3, random_state=109)
    print(X.shape)
    print(Y.shape)
    if classifier == "SVM":
        clf = svm.SVC(kernel='linear', C=10, cache_size=7000)
    elif classifier == "LR":
        clf = LogisticRegression()
    
    elif classifier == "RF":
        class_weight = dict({1:1,0:25})
        clf = RandomForestClassifier(bootstrap=True,
                class_weight=class_weight,
                    criterion='gini',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=4, min_samples_split=10,
                    min_weight_fraction_leaf=0.0, n_estimators=300,
                    oob_score=False,
                    random_state=0,
                    verbose=0, warm_start=False)

    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    _, f, o = helper.getResult(y_test, y_pred)
    print("                                Overall               #               Fake                ")
    print("                   precision    recall      f1-score  #  precision    recall      f1-score")
    res = helper.printResult(exp,o,f)
    print(res)
    if save:
        path = args.model+"_results.txt"
        helper.saveResults(path, res)
    #Save Model
    outfile = open(config.API+'SVM.pkl', 'wb')
    pickle.dump(clf, outfile)
    outfile.close()

    



parser = argparse.ArgumentParser(description='Argparse!')
subparsers = parser.add_subparsers()

parser_p = subparsers.add_parser('Unigram')
parser_p.set_defaults(func=unigram)

parser_q = subparsers.add_parser('Bigram')
parser_q.set_defaults(func=bigram)

parser_p = subparsers.add_parser('Trigram')
parser_p.set_defaults(func=trigram)

parser_q = subparsers.add_parser('U+B+T')
parser_q.set_defaults(func=u_b_t)


parser_p = subparsers.add_parser('C3-gram')
parser_p.set_defaults(func=char_3)

parser_q = subparsers.add_parser('C4-gram')
parser_q.set_defaults(func=char_4)

parser_p = subparsers.add_parser('C5-gram')
parser_p.set_defaults(func=char_5)

parser_q = subparsers.add_parser('C3+C4+C5')
parser_q.set_defaults(func=char_3_4_5)


parser_p = subparsers.add_parser('Lexical')
parser_p.set_defaults(func=lexical)

parser_q = subparsers.add_parser('POS')
parser_q.set_defaults(func=pos)

parser_q = subparsers.add_parser('L_POS')
parser_q.set_defaults(func=L_POS)

parser_p = subparsers.add_parser('Emb_F')
parser_p.set_defaults(func=word_300)

parser_q = subparsers.add_parser('Emb_N')
parser_q.set_defaults(func=word_100)


parser_p = subparsers.add_parser('L+POS+E_F')
parser_p.set_defaults(func=L_POS_Emb_F)

parser_q = subparsers.add_parser('L+POS+E_N')
parser_q.set_defaults(func=L_POS_Emb_N)

parser_p = subparsers.add_parser('MP')
parser_p.set_defaults(func=mp)

parser_q = subparsers.add_parser('L+POS+E_F+MP')
parser_q.set_defaults(func=L_POS_Emb_F_MP)

parser_q = subparsers.add_parser('L+POS+E_N+MP')
parser_q.set_defaults(func=L_POS_Emb_N_MP)


parser_p = subparsers.add_parser('all_features')
parser_p.set_defaults(func=allfeatures)

parser.add_argument("model")
parser.add_argument("-s","--save", action="store_true")
args = parser.parse_args()
classifier = args.model
save = args.save
X = load_data()
Y = X[["label"]]

args.func(args)









