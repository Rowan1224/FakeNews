import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import scale
from collections import Counter

Featurelist = []
df = pd.read_csv("../Data/Corpus/AllDataFeature.csv",)

headline = ["nwh","nsh","nph","nesh"]
article = ["nwa","nsa","avg_wps","npa","nesa","mfwa","lfwa","nsw"]
domain = ["rank"]
Allfeatures = ["nwh","nwa","nsh","nsa","avg_wps","nph","npa","nesh","nesa","mfwa","lfwa","nsw"]
readability = ["CLI"]
path = "../Data/result/SVM-All-Results1.txt"
with open(path ,"w") as outfile:
    for feature in readability:
        print(feature)
        df = pd.read_csv("../Data/Corpus/AllDataFeature.csv", )
        X = df[[feature]]
        Y = df[["label"]]
        print(Y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, Y.values.ravel(), test_size=0.3, random_state=109)
        clf = svm.SVC(kernel='linear',C=10,cache_size=7000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        outfile.write(feature+"\n")
        outfile.write("Train: "+str(Counter(y_train))+"\n")
        outfile.write("Accuracy: "+str(metrics.accuracy_score(y_test, y_pred))+"\n")
        outfile.write("Precision: "+str(metrics.precision_score(y_test, y_pred))+"\n")
            # Model Recall: what percentage of positive tuples are labelled as such?
        outfile.write("Recall:"+str(metrics.recall_score(y_test, y_pred))+"\n")
        outfile.write("F1-Score: "+str(metrics.f1_score(y_test, y_pred))+"\n")
        outfile.write("Confusion Matrix: "+str(metrics.confusion_matrix(y_test,y_pred))+"\n")
        outfile.write(metrics.classification_report(y_test,y_pred)+"\n")
        outfile.write("===================================================================\n")
