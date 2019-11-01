import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler,scale
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier

Allfeatures = ["mean"]
CSV = "../Data/Corpus/word-embedding-FT-Alldata.csv"
path = "../Data/result/SVM-Word-Embedding-Results-FT-SUM.txt"
with open(path, "w") as outfile:
    for feature in Allfeatures:
        CSV = "../Data/Corpus/word-embedding-FT-SUM-Alldata.csv"
        CSV2 = "../Data/Corpus/word-embedding-Alldata.csv"
        df1 = pd.read_csv(CSV, )
        df = df1.dropna()
        X = df[[feature]]
        Y = df[["label"]]
        rus = RandomUnderSampler(random_state=0)
        X_resample, y_resample = rus.fit_sample(X, Y)
        print(df.label.value_counts())
        X_train, X_test, y_train, y_test = train_test_split(X, Y.values.ravel(), test_size=0.3, random_state=109)
        clf = svm.SVC(kernel='linear',C=10,cache_size=7000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(Counter(y_pred))

        # outfile.write("Train: "+str(Counter(y_train))+"\n")
        outfile.write("Accuracy: "+str(metrics.accuracy_score(y_test, y_pred))+"\n")
        outfile.write("Precision: "+str(metrics.precision_score(y_test, y_pred))+"\n")
        outfile.write("Recall:"+str(metrics.recall_score(y_test, y_pred))+"\n")
        outfile.write("F1-Score: "+str(metrics.f1_score(y_test, y_pred))+"\n")
        outfile.write("Confusion Matrix: "+str(metrics.confusion_matrix(y_test,y_pred))+"\n")
        outfile.write(metrics.classification_report(y_test,y_pred)+"\n")

