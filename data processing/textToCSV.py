import csv
import os
import pandas as pd


def docTolist_V1(path,articleID,label, f_type):
    data = [articleID]
    with open(path, "r") as infile:
        for line in infile:
            line = line.replace("\n", "")
            data.append(line)
    if len(data) == 9:
        data.remove(data[4])
        data.append(label)
        data.append(f_type)
        return data
    else:
        print(path)
        return [articleID]


def docTolist_V2(path, articleID, label):
    data = [articleID]
    temp = []
    with open(path, "r") as infile:
        for line in infile:
            line = line.replace("\n", "")
            temp.append(line)

    if len(temp) == 6:
        data.append(temp[1])
        data.append(temp[0])
        data.append(temp[2])
        data.append(temp[3])
        data.append("Related")
        data.append(temp[4])
        data.append(temp[5])
        data.append(label)
        return data
    else:
        print(path)
        return [articleID]


def text2csv():
    V1base = "/home/rowan/PycharmProjects/FakeNews/Data/Dataset/AllData"
    V2base = "/media/MyDrive/Project Fake news/Data/new data/Rifat/Data_1"
    V1 = os.listdir(V1base)
    V2 = os.listdir(V2base)

    with open('../Data/Corpus/Corpus.csv', 'w') as CSV:
        writer = csv.writer(CSV)
        headers = ["articleID", "domain", "date", "type", "source", "relation", "headline", "content", "label"]
        writer.writerow(headers)
        articleID = 0
        for doc in V1:
            path = V1base + "/" + doc
            articleID += 1
            label = 0
            if doc.split("-")[0] == "True":
                label = 1
            writer.writerow(docTolist_V1(path, str(format(articleID, '04d')), label))

        for doc in V2:
            path = V2base + "/" + doc
            articleID += 1
            label = 1
            writer.writerow(docTolist_V2(path, str(format(articleID, '04d')), label))

def text2csvV2():
    V1base = "/home/rowan/PycharmProjects/FakeNews/Data/Dataset/C+S+F"

    V1 = os.listdir(V1base)

    with open('../Data/Corpus/FakeCorpus.csv', 'w') as CSV:
        writer = csv.writer(CSV)
        headers = ["articleID", "domain", "date", "type", "source", "relation", "headline", "content", "label", "F-type"]
        writer.writerow(headers)
        articleID = 0
        for doc in V1:
            path = V1base + "/" + doc
            articleID += 1
            label = 0
            f_type = doc.split("-")[0]
            writer.writerow(docTolist_V1(path, str(format(articleID, '04d')), label, f_type))


def addNewData():
    df = pd.read_csv("../Data/Corpus/Corpus.csv")
    series = []
    filepath = "path"
    filelist = os.listdir(filepath)
    # for doc in filelist:


text2csvV2()
df = pd.read_csv("../Data/Corpus/FakeCorpus.csv")
print(df.shape)
df = df.drop_duplicates(subset="content", keep=False)
df.to_csv("../Data/Corpus/FakeCorpus.csv")
print(df.shape)

