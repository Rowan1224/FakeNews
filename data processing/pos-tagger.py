# # import spacy
# # # import bn_core_news_sm
# #
# # nlp = spacy.load("/media/MyDrive/Project Fake news/Models/bn_core_news_sm-2.0.0/bn_core_news_sm/bn_core_news_sm-2.0.0")
# # doc = nlp(u'আমি বাংলায় গান গাই। তুমি কি গাও?')
# #
# #
# # print(doc.ents)
# # for token in doc:
# #     print(token.text, token.pos_)
#
#
# # nlp = spacy.load("en_core_web_sm")
# # doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
# #
# # for token in doc:
# #     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
# #             token.shape_, token.is_alpha, token.is_stop)
#


import pandas as pd
import requests, json

path = ""
Baseurl = "http://10.100.222.160:5004/predict"


def doc2senlist(doc):

    doc = doc.replace("?", "!")
    doc = doc.replace("।", "!")
    sentenceList = doc.split("!")

    return sentenceList


def pos(sen):
    param ={"sentence": sen}
    response = requests.get(url=Baseurl, params=param).json()
    result = json.loads(response)
    return result


def counter(posCount, posResult):
    for token, pos in posResult:
        if pos in posCount.keys():
            posCount[pos] = posCount[pos] + 1
        else:
            posCount[pos] = 1

    return posCount


df = pd.read_csv(path)

for row in df.iterrows():
    news = row[0][2]
    sentenceList = doc2senlist(news)
    posCount = {}
    for sentence in sentenceList:
        posResult = pos(sentence)
        posCount = counter(posCount, posResult)

    print(posCount)
