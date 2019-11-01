import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler,scale
from collections import Counter

# Create an empty list
Featurelist = []
df = pd.read_csv("../Data/Corpus/AllDataFeature.csv",)
headline = ["nwh","nsh","nph","nesh"]
article = ["nwa","nsa","avg_wps","npa","nesa","mfwa","lfwa","nsw"]
domain = ["rank"]
readability = ["CLI","ARI"]
Allfeatures = ["nwh","nwa","nsh","nsa","avg_wps","nph","npa","nesh","nesa","mfwa","lfwa","rank","nsw","CLI","ARI"]
path = "../Data/result/LR-All-Results.txt"
with open(path ,"w") as outfile:
    for feature in Allfeatures:
        df = pd.read_csv("../Data/Corpus/AllDataFeature.csv", )
        df = df.dropna()
        X = df[[feature]]
        Y = df[["label"]]
        print(Y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, Y.values.ravel(), test_size=0.3, random_state=109)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(Counter(y_pred))
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

        # report = metrics.classification_report(y_test, y_pred, output_dict=True)
        # df = pd.DataFrame(report).transpose()
        # df.to_excel("Experiment_3_LR.xlsx")