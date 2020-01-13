from collections import Counter

import pandas as pd
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def printScore(y_test, y_pred):
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))

    print("F1-Score:", metrics.f1_score(y_test, y_pred))

    print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))

    print(metrics.classification_report(y_test, y_pred, output_dict=True))
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    reportList = [report["0"]]
    df = pd.DataFrame(reportList)
    df.to_csv("../Data/result/try.csv", index=None, header=True)

def experiment(path, features, df, Y):

    with open(path, "w") as outfile:
        for feature in features:

            X = df[[feature]]

            print(Y.shape)

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109)
            # clf = LogisticRegression()
            clf = svm.SVC(kernel='linear', C=10, cache_size=7000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print(feature)
            printScore(y_test,y_pred)
            outfile.write(feature + "\n")
            outfile.write("Train: " + str(Counter(y_train)) + "\n")
            outfile.write("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")
            outfile.write("Precision: " + str(metrics.precision_score(y_test, y_pred)) + "\n")
            # Model Recall: what percentage of positive tuples are labelled as such?
            outfile.write("Recall:" + str(metrics.recall_score(y_test, y_pred)) + "\n")
            outfile.write("F1-Score: " + str(metrics.f1_score(y_test, y_pred)) + "\n")
            outfile.write("Confusion Matrix: " + str(metrics.confusion_matrix(y_test, y_pred)) + "\n")
            outfile.write(metrics.classification_report(y_test, y_pred) + "\n")
            outfile.write("===================================================================\n")


df = pd.read_csv("../Data/Corpus/AllPOSDataNLTK_N_MXMN.csv")
df1 = pd.read_csv("../Data/Corpus/AllDataTarget.csv")
df = df.drop(['articleID', 'Unnamed: 0'], axis=1)
head = list(df)
print(head)
df = df.fillna(0)
Y = df1.label.values
path = "../Data/result/LR/LR-All-Results-POS-all-Nltk-N_MXMN.txt"

X = df[head]

print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109)
# clf = LogisticRegression()
clf = svm.SVC(kernel='linear', C=10, cache_size=7000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

printScore(y_test, y_pred)


# with open(path, "w") as outfile:
#
#     X = df[head]
#
#     print(Y.shape)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109)
#     # clf = LogisticRegression()
#     clf = svm.SVC(kernel='linear', C=10, cache_size=7000)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#
#     printScore(y_test, y_pred)

    # outfile.write("Train: " + str(Counter(y_train)) + "\n")
    # outfile.write("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")
    # outfile.write("Precision: " + str(metrics.precision_score(y_test, y_pred)) + "\n")
    #     # Model Recall: what percentage of positive tuples are labelled as such?
    # outfile.write("Recall:" + str(metrics.recall_score(y_test, y_pred)) + "\n")
    # outfile.write("F1-Score: " + str(metrics.f1_score(y_test, y_pred)) + "\n")
    # outfile.write("Confusion Matrix: " + str(metrics.confusion_matrix(y_test, y_pred)) + "\n")
    # outfile.write(metrics.classification_report(y_test, y_pred) + "\n")
    # outfile.write("===================================================================\n")