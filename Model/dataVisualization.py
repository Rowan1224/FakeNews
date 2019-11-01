import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
import researchpy as rp

TrueData = []
FakeData = []
ExpData = []
csvheaderlist = ["nwh","nwa","nsh","nsa","avg_wps","nph","npa","nesh","nesa","mfwa","lfwa","nsw"]

def check():
    df2 = pd.read_csv("Data/Corpus/ExpFeature.csv")

    for head in csvheaderlist:
        print()

def dataload():
    Featurelist = []
    df = pd.read_csv("Data/Corpus/NewFeature.csv")
    print(df.shape)
    # Iterate over each row
    for index, rows in df.iterrows():
        Featurelist.append(list(rows))

    # # Print the list
    # print(Row_list)
    #
    # print(lines)
    df1 = pd.read_csv("Data/Corpus/NewTarget.csv")
    saved_column = df1.label
    print(list(saved_column))

    df['target'] = list(saved_column)
    df2 =  pd.read_csv("Data/Corpus/ExpFeature.csv")

    for head in csvheaderlist:
        print(head)
        TrueData.append(rp.summary_cont(df.groupby('target')[head])["Mean"][1])
        FakeData.append(rp.summary_cont(df.groupby('target')[head])["Mean"][0])
        ExpData.append(rp.summary_cont(df2[head])["Mean"][0])

    print(TrueData)
    print(FakeData)
    print(ExpData)
# data to plot
def plotStatisticalData():
    n_groups = len(csvheaderlist)
    means_true = tuple(TrueData)
    means_fake = tuple(FakeData)
    plt.rcParams["figure.figsize"] = [20, 15]
    plt.rcParams.update({'font.size': 25})
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, means_true, bar_width,
                     alpha=opacity,
                     color='C0',
                     label='True')

    rects2 = plt.bar(index + bar_width, means_fake, bar_width,
                     alpha=opacity,
                     color='C1',
                     label='Fake')

    plt.xlabel('Stylistic Feature')
    plt.ylabel('Mean')
    plt.title('Statistical Comparison')
    plt.xticks(index + bar_width/2, tuple(csvheaderlist))
    plt.legend()

    plt.tight_layout()
    plt.savefig('stat2.png')

def result():
    n_groups = 3
    means_true = (82.72884283246977,80.00,88.59060402684564)
    means_fake = (86.01036269430051,82.77945619335347,91.94630872483222)
    means_true_LR = (82.21070811744386, 79.27927927927928, 88.59060402684564)
    means_fake_LR = (84.97409326424871, 81.68168168168168, 91.2751677852349)
    label = ("Accuracy", "Precision", "Recall")
    # create plot
    plt.rcParams["figure.figsize"] = [36, 27]
    plt.rcParams.update({'font.size': 60})
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, means_true_LR, bar_width,
                     alpha=opacity,
                     color='C0',
                     label='Stylistic Feature')

    rects2 = plt.bar(index + bar_width, means_fake_LR, bar_width,
                     alpha=opacity,
                     color='C1',
                     label='Stylistic Features & alexa Ranking')

    plt.xlabel('')
    plt.ylabel('Score')
    plt.title('LR Results')
    plt.xticks(index + bar_width/2, label)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig('LR-Results.png')

def genTable():
    with open("Data/Corpus/table.csv", 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(["Feature","True","Fake","Exp"])
        for i in range(0,len(csvheaderlist)):
            row = []
            row.append(csvheaderlist[i])
            row.append(TrueData[i])
            row.append((FakeData[i]))
            row.append((ExpData[i]))
            writer.writerow(row)

def plotSVM():
    Featurelist = []
    df = pd.read_csv("Data/Corpus/NewFeature.csv", usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    print(df.shape)
    # Iterate over each row
    for index, rows in df.iterrows():
        Featurelist.append(list(rows))

    df = pd.read_csv("Data/Corpus/NewTarget.csv")
    saved_column = df.label
    print(list(saved_column))

    X_train, X_test, y_train, y_test = train_test_split(Featurelist, list(saved_column), test_size=0.3,
                                                        random_state=109)
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear', C=10, cache_size=7000)  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train)
    pca = PCA(n_components=2).fit(X_train)
    pca_2d = pca.transform(X_train)
    for i in range(0, pca_2d.shape[0]):
        c1 , c2
        if y_train[i] == 0:
            c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif y_train[i] == 1:
            c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')

        plt.legend([c1, c2], ['False', 'Versicolor', 'Virginica'])
    plt.title('Iris training dataset with 3 classes and    known outcomes')
    plt.show()


dataload()
result()
# genTable()

#check()
