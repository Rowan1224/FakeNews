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


def classifier(y_test,y_pred):

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1-Score:", metrics.f1_score(y_test, y_pred))
    print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


df = pd.read_csv("../Data/Corpus/AllDataTarget.csv", )
featureVector = []
labels = []
for row in df.iterrows():
    row = row[1]
    label = row["label"]
    labels.append(label)

#
y_train, y_test = train_test_split(labels, test_size=0.3, random_state=109)
y_major = []
y_random = []

for i in range(len(y_test)):
    a = 1
    b = random.choice([0, 1])
    # print(b)
    y_major.append(a)
    y_random.append(b)

print("Major Baseline")
classifier(y_test, y_major)
print("Random Baseline")
classifier(y_test, y_random)