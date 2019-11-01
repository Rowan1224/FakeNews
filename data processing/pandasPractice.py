import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression
from pandas import ExcelWriter
from pandas import ExcelFile






def result(figureTitle,results,data,model):
    n_groups = 3
    label = ("F1-Score", "Precision", "Recall")
    # create plot
    plt.rcParams["figure.figsize"] = [36, 90]
    plt.rcParams.update({'font.size': 62})
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, results[0], bar_width,
                     alpha=opacity,
                     color='C0',
                     label=data[0][0]+' VS '+data[0][1])

    rects2 = plt.bar(index + bar_width, results[1], bar_width,
                     alpha=opacity,
                     color='C1',
                     label=data[1][0]+' VS '+data[1][1])

    rects3 = plt.bar(index + 2*bar_width, results[2], bar_width,
                     alpha=opacity,
                     color='C2',
                     label=data[2][0]+' VS '+data[2][1])

    plt.xlabel('')
    plt.ylabel('Score')
    plt.title(model+' Results for Fake data')
    plt.xticks(index + bar_width/2, label)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(figureTitle)


csvheaderlist = ["articleID","nwh","nwa","nsh","nsa","avg_wps","nph","npa","nesh","nesa","mfwa","lfwa","rank","nsw","CLI","ARI","label"]

results = []
data = []
model = "SVM"
T = 0
for i in range(1,4):
    df = pd.read_excel("Data/result/Experiment_"+str(i)+"_"+model+".xlsx")
    r = (df["f1-score"][T],df["precision"][T],df["recall"][T])
    results.append(r)
    d = (str(df["Train"][0]),str(df["Train"][1]))
    data.append(d)

result("SVM1",results,data,model)

# Statistics
# head.remove("articleID")
# head.remove("label")
# stat = df.groupby('label').mean()
# stat.to_excel("FeatureStat.xlsx")

# selectKBestFeature
# print(df[df > 0])
# X = df[["nwh","nwa","nsh","nsa","avg_wps","nph","npa","nesh","nesa","mfwa","lfwa","rank","nsw","CLI","ARI"]]
# Y = df[["label"]]
# selector = SelectKBest(f_regression, k=10)
# X_new = selector.fit_transform(X, Y)
# vector_names = list(X.columns[selector.get_support(indices=True)])
# print(vector_names)

