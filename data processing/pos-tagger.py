# %%

import pandas as pd
import requests, json
path = r"C:\Users\user\Desktop\tomorrow\Corpus\AllDataTarget.csv" #csv file path
Baseurl = "http://10.100.222.160:5004/predict"


def doc2senlist(doc):
    doc = doc.replace("?", "!")
    doc = doc.replace("।", "!")
    sentenceList = doc.split("!")

    return sentenceList


def pos(sen):
    param = {"sentence": sen}
    response = requests.get(url=Baseurl, params=param).json()

    return response


def POScounter(posCount, posResult):
    for token, pos in posResult.items():
        if pos in posCount.keys():
            posCount[pos] = posCount[pos] + 1
        else:
            posCount[pos] = 1

    return posCount


df = pd.read_csv(path)
dataset = []
header = set()
for row in df.iterrows():

    news = row[1]["news"]
    sentenceList = doc2senlist(news)
    posCount = {}
    posCount["articleID"] = row[1]["articleID"]
    for sentence in sentenceList:
        posResult = pos(sentence)
        posCount = POScounter(posCount, posResult)
    #     for head in posCount.keys():
    #         header.add(head)
    dataset.append(posCount)
    print(len(dataset))
#     print(posCount)

print(header)
print(len(dataset))

output = pd.DataFrame.from_dict(dataset)
output.to_csv(r"C:\Users\user\Desktop\tomorrow\Corpus\AllPOSData.csv")

# %%


