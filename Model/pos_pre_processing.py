
import nltk
from nltk.corpus import indian
from nltk.tag import tnt
import string
import pandas as pd

nltk.download('indian')
nltk.download('punkt')

tagged_set = 'bangla.pos'
word_set = indian.sents(tagged_set)
count = 0
for sen in word_set:
    count = count + 1
    sen = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in sen]).strip()
    # print (count, sen)
print ('Total sentences in the tagged file are',count)

train_perc = .96

train_rows = int(train_perc*count)
test_rows = train_rows + 1

data = indian.tagged_sents(tagged_set)
train_data = data[:train_rows] 
test_data = data[test_rows:]
pos_tagger = tnt.TnT()
pos_tagger.train(train_data)
pos_tagger.evaluate(test_data)


path = "../Data/Corpus/AllDataTarget.csv"
Baseurl = "http://10.100.222.160:5004/predict"


def doc2senlist(doc):

    doc = doc.replace("?", "!")
    doc = doc.replace("।", "!")
    sentenceList = doc.split("!")

    return sentenceList


def pos(sen):
    response = {}
    tokenized = nltk.word_tokenize(sen)
    try:
        result = pos_tagger.tag(tokenized)
        for r in result:
            # print(r)
            token, pos = r
            response[token] = pos
    except:
        print(sen)

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
    if len(dataset) % 100 == 0:
        print(len(dataset))
    

output = pd.DataFrame.from_dict(dataset)
print(output.shape)
normalized_df=(output-output.mean())/output.std()
normalized_df.to_csv("../Data/Corpus/AllPOSDataNLTK_N_STD.csv")
normalized_df=(output-output.min())/(output.max()-output.min())
normalized_df.to_csv("../Data/Corpus/AllPOSDataNLTK_N_MXMN.csv")



