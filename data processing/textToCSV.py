import csv
import os
import pandas as pd

df = pd.read_csv("../Data/Corpus/AuthenticCorpus.csv")
df = df[['headline', 'category']]
headlineCategory = {}
for row in df.iterrows():
    h = row[1]['headline']
    c = row[1]['category']
    headlineCategory[h] = c

v1base = "/home/rowan/PycharmProjects/FakeNews/Data/Labeled Dataset/AllData/V1"
v2base = "/home/rowan/PycharmProjects/FakeNews/Data/Labeled Dataset/AllData/V2"
v1 = os.listdir(v1base)
v2 = os.listdir(v2base)


def doc_to_list_v1(path, articleID, label):
    data = [articleID]
    temp = []

    with open(path, "r") as infile:
        try:
            for line in infile:
                line = line.replace("\n", "")
                line = str(line).strip()
                temp.append(line)
        except:
            print("v1 code " + path.replace(v1base, ""))

    if len(temp) == 6:

        if temp[4] in headlineCategory:
            category = headlineCategory[temp[4]]
        else:

            category = "National"
        data.append(temp[0].replace(" ", ""))
        data.append(temp[1])
        data.append(category)
        data.append(temp[3])
        data.append("Related")
        data.append(temp[4])
        data.append(temp[5])
        data.append(label)
        if len(data) != 9:
            print("v1 code " + path.replace(v1base, ""))
        if not temp[3]:
            print("v1 6 " + path.replace(v1base, ""))

        return data
    elif len(temp) == 8:

        for i in range(8):
            # if i == 0:
            #     data.append(temp[i].replace(" ", ""))
            if i == 2:
                continue
            elif i == 3:
                if temp[6] in headlineCategory:
                    category = headlineCategory[temp[6]]
                else:
                    category = "National"
                data.append(category)
            else:
                if not temp[i]:
                    print("v1 8 " + path.replace(v1base, ""))
                d = temp[i].lstrip()
                data.append(d)

        data.append(label)
        if len(data) != 9:
            print("v1 code " + path.replace(v1base, ""))

        return data

    else:
        print("v1 " + path.replace(v1base, ""))
        return [articleID]


def doc_to_list_v2(path, articleID, label):
    data = [articleID]
    temp = []
    with open(path, "r") as infile:
        try:
            for line in infile:
                line = str(line).strip()
                line = line.replace("\n", "")
                temp.append(line)
        except:
            print("v2 code " + path.replace(v2base, ""))

    if len(temp) == 6:
        if temp[4] in headlineCategory:
            category = headlineCategory[temp[4]]
        else:
            category = "National"

        # if temp[1].replace(".", "").isalpha():
        #     domain = temp[1]
        #     date = temp[0]
        # else:
        #     domain = temp[0]
        #     date = temp[1]

        data.append(temp[1].replace(" ", ""))
        data.append(temp[0])
        data.append(category)
        data.append(temp[3])
        data.append("Related")
        data.append(temp[4])
        data.append(temp[5])
        data.append(label)

        if len(data) != 9:
            print("v2 6 " + path.replace(v2base, ""))

        if not temp[3]:
            print("v2 6 " + path.replace(v2base, ""))

        return data
    elif len(temp) == 8:
        for i in range(8):
            # if i == 0:
            #     data.append(temp[i].replace(" ", ""))
            if i == 2:
                continue
            elif i == 3:
                if temp[6] in headlineCategory:
                    category = headlineCategory[temp[6]]
                else:
                    category = "National"
                data.append(category)
            else:
                if not temp[i]:
                    print("v2 8 " + path.replace(v2base, ""))
                d = temp[i].lstrip()
                data.append(d)

        data.append(label)

        if len(data) != 9:
            print("v2 9 " + path.replace(v2base, ""))

        return data

    else:

        print("v2 "+path.replace(v2base, ""))
        return [articleID]


def text_to_csv():

    with open('../Data/Corpus/LabeledAuthenticCorpus.csv', 'w') as CSV:
        writer = csv.writer(CSV)
        headers = ["articleID", "domain", "date", "category", "source", "relation", "headline", "content", "label"]
        writer.writerow(headers)
        articleID = 0
        for doc in v1:
            path = v1base + "/" + doc
            articleID += 1
            label = 0
            if doc.split("-")[0] == "True":
                label = 1
            writer.writerow(doc_to_list_v1(path, str(format(articleID, '04d')), label))

        for doc in v2:
            path = v2base + "/" + doc
            articleID += 1
            label = 1
            writer.writerow(doc_to_list_v2(path, str(format(articleID, '04d')), label))




def addNewData():
    df = pd.read_csv("../Data/Corpus/Corpus.csv")
    series = []
    filepath = "path"
    filelist = os.listdir(filepath)
    # for doc in filelist:


# text_to_csv_v2()
# df = pd.read_csv("../Data/Corpus/FakeCorpus.csv")
# print(df.shape)
# df = df.drop_duplicates(subset="content", keep="First")
# df.to_csv("../Data/Corpus/FakeCorpus.csv")
# print(df.shape)

text_to_csv()
# text_to_csv_fake()

