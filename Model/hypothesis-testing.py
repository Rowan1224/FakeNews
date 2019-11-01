import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import researchpy as rp


# Create an empty list
Featurelist = []
df = pd.read_csv("../Data/Corpus/Feature.csv")
print(df.shape)
# Iterate over each row
for index, rows in df.iterrows():
    Featurelist.append(list(rows))

# # Print the list
# print(Row_list)
#
# print(lines)
df1 = pd.read_csv("../Data/Corpus/Target.csv")
saved_column = df1.label
print(list(saved_column))

df['target']= list(saved_column)
TrueData = []
FakeData = []
csvheaderlist = ["nwh","nwa","nsh","nsa","avg_wps","nph","npa","nesh","nesa","mfwa","lfwa"]
for head in csvheaderlist:
    print(head)
    TrueData.append(rp.summary_cont(df.groupby('target')[head])["Mean"][0])
    FakeData.append(rp.summary_cont(df.groupby('target')[head])["Mean"][1])


print(TrueData)
print(FakeData)