import string, re, os
import unicodedata
import pandas as pd
import requests
from bs4 import BeautifulSoup

puncList = ["।", "”", "“", "’"]
for p in string.punctuation.lstrip():
    puncList.append(p)


def sentence_count(doc):

    try:
        doc = doc.replace("ShareTweet", "")
        doc = doc.replace("Take Our Poll", "")
    except:
        print(str(doc))
    doc = doc.replace("?", "!")
    doc = doc.replace("।", "!")
    sentencelist = doc.split("!")

    return sentencelist


def cleanword(word):
    for p in puncList:
        word = word.replace(p, "")
    word = re.sub(r'[\u09E6-\u09EF]', "", word, re.DEBUG)  # replace digits

    return word


def count_chars(tokens):
    count = 0
    for word in tokens:
        word = cleanword(word)
        try:
            l = list(map(unicodedata.name, word))
            count += len(l)
        except:
            print(word)

    return count


def punctuation_counter(sentence_list):

    total = 0
    for sen in sentence_list:
        char_list = str(sen).lstrip()
        for char in char_list:
            if char in puncList:
                total += 1

    return total


def tokenizer(doc):

    tokens = []

    for word in doc.split(" "):
        word = cleanword(word)
        if word != "":
            tokens.append(word)

    return tokens


def analyser(content):
    sentence_list = sentence_count(content)
    total_sentence = len(sentence_list) - 1
    total_punctuation = punctuation_counter(sentence_list) + total_sentence
    tokens = tokenizer(content)
    total_word = len(tokens)
    total_chars = count_chars(tokens)

    return total_chars, total_punctuation, total_word, total_sentence


def true_news(df_true):
    df_true = df_true.dropna()
    chars = 0
    punc = 0
    word = 0
    sentence = 0
    for row in df_true.iterrows():
        content = row[1]["content"]
        c, p, w, s = analyser(content)
        chars += c
        punc += p
        word += w
        sentence += s
    print(df_true.shape)
    print("Type         True")
    print("Chars         " + str(chars))
    print("Punctuation         " + str(punc))
    print("Word         " + str(word))
    print("Sentence         " + str(sentence))
    print("===================================================")


def alexa_rank(domain):
    url = 'https://www.alexa.com/siteinfo/'+domain
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    for soup in soup.find_all(class_='rankmini-rank'):
        rank = str(soup.getText()).replace("\n", "")
        rank = rank.replace("\t", "")
        rank = rank.replace(" ", "")
        rank = rank.replace("#", "")
        rank = rank.replace(",", "")
        return rank


def fake_news():
    dir = os.listdir("../Data/Dataset/Fake News")
    base = "../Data/Dataset/Fake News/"
    listOfDict = []
    for news in dir:
        with open(base+news,"r") as infile:
            i = 0
            row = {}
            for line in infile:

                i += 1
                if i == 2:
                    row["domain"] = str(line).replace("\n", "")
                if i == 5:
                    row["content"] = str(line).replace("\n", "")
        listOfDict.append(row)

    df_fake = pd.DataFrame(listOfDict)
    df_fake = df_fake.dropna()
    domain_count = df_fake.domain.value_counts().to_dict()

    print(len(domain_count))
    chars = 0
    punc = 0
    word = 0
    sentence = 0
    for row in df_fake.iterrows():
        content = row[1]["content"]
        c, p, w, s = analyser(content)
        chars += c
        punc += p
        word += w
        sentence += s
    print(df_fake.shape)
    print("Type         Fake")
    print("Chars         " + str(chars))
    print("Punctuation         " + str(punc))
    print("Word         " + str(word))
    print("Sentence         " + str(sentence))
    print("===================================================")

    return domain_count


def appendix():
    df_true = pd.read_csv("../Data/Corpus/RowCorpus.csv")
    df_fake = pd.read_csv("../Data/Corpus/FakeCorpus.csv")
    true = df_true.domain.value_counts().to_dict()
    fake = fake_news()
    others = df_fake.groupby("F-type").domain.value_counts().to_dict()
    satire = {}
    clickbait = {}
    for key, value in others.items():
        t, d = key
        if t == "Clickbaits":
            clickbait[d] = value
        else:
            satire[d] = value
    print(satire)

    trueList = []
    for key, value in true.items():
        row = {}
        domain = key
        count = value
        rank = alexa_rank(domain)
        row["domain"] = domain
        row["rank"] = rank
        row["count"] = count
        trueList.append(row)

    FakeList = []
    for key, value in fake.items():
        row = {}
        domain = key
        count = value
        rank = alexa_rank(domain)
        row["domain"] = domain
        row["rank"] = rank
        row["count"] = count
        FakeList.append(row)

    satireList = []
    for key, value in satire.items():
        row = {}
        domain = key
        count = value
        rank = alexa_rank(domain)
        row["domain"] = domain
        row["rank"] = rank
        row["count"] = count
        satireList.append(row)

    clickbaitList = []
    for key, value in clickbait.items():
        row = {}
        domain = key
        count = value
        rank = alexa_rank(domain)
        row["domain"] = domain
        row["rank"] = rank
        row["count"] = count
        clickbaitList.append(row)

    pd.DataFrame(trueList).to_csv("../Data/Appendix/True.csv")
    pd.DataFrame(FakeList).to_csv("../Data/Appendix/Fake.csv")
    pd.DataFrame(satireList).to_csv("../Data/Appendix/Satire.csv")
    pd.DataFrame(clickbaitList).to_csv("../Data/Appendix/Clickbait.csv")

# df_true = pd.read_csv("../Data/Corpus/RowCorpus.csv")
# df_fake = pd.read_csv("../Data/Corpus/FakeCorpus.csv")
#
# types_of_fake_news = df_fake["F-type"].unique()
#
# print(df_fake["F-type"].value_counts())
# for T in types_of_fake_news:
#     chars = 0
#     punc = 0
#     word = 0
#     sentence = 0
#     total = df_fake["F-type"].value_counts()
#     for row in df_fake.iterrows():
#         content = row[1]["content"]
#         Type = row[1]["F-type"]
#         if Type != T:
#             continue
#         c, p, w, s = analyser(content)
#         chars += c
#         punc += p
#         word += w
#         sentence += s
#
#     print("Type         "+T)
#     print("Chars         " + str(chars))
#     print("Punctuation         " + str(punc))
#     print("Word         " + str(word))
#     print("Sentence         " + str(sentence))
#     print("===================================================")
#
# true_news(df_true)

fake_news()