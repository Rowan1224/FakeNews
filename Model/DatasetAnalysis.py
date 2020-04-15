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


def df_iteration(df, type):
    df_true = df.dropna()
    total = df_true.shape[0]
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
    print("Type         "+type)
    print("Chars         " + str(chars/total))
    print("Punctuation         " + str(punc/total))
    print("Word         " + str(word/total))
    print("Sentence         " + str(sentence/total))
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


def appendix():
    df_true = pd.read_csv("../Fake News Dataset/Authentic-48K.csv")
    df_fake = pd.read_csv("../Fake News Dataset/LabeledFake-1K.csv")
    true = df_true.domain.value_counts().to_dict()
    others = df_fake.groupby("F-type").domain.value_counts().to_dict()
    fake = {}
    satire = {}
    clickbait = {}
    total_true = 0
    total_fake = 0
    total_satire = 0
    total_clickbait = 0
    for key, value in others.items():
        t, d = key
        if t == "Clickbaits":
            clickbait[d] = value
        elif t == "Satire":
            satire[d] = value
        else:
            fake[d] = value
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
        total_true += count
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
        total_fake += count
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
        total_satire += count
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
        total_clickbait += count
        clickbaitList.append(row)

    print(total_true)
    print(total_fake)
    print(total_satire)
    print(total_clickbait)
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

# fake_news()

# path = "../Fake News Dataset/LabeledFake-1K.csv"
#
# df = pd.read_csv(path)
# print(list(df))
# df = df["F-type"].value_counts()
# fake = dict(df)
# print(fake)
# final = {}
# # for k, v in fake.items():
# #     final[k] = fake[k]+true[k]
# #     print(k+" "+str(final[k]))

def categories():
    path = "../Fake News Dataset/Authentic-48K.csv"
    df = pd.read_csv(path)
    print(df.shape)
    df = df.dropna()
    df = df["category"].value_counts()
    true = dict(df.dropna())
    print(true)

    path = "../Fake News Dataset/LabeledFake-1K.csv"
    df = pd.read_csv(path)
    df = df["category"].value_counts()
    false = dict(df.dropna())
    print(false)


def dataDistribution():
    df_true = pd.read_csv("../Fake News Dataset/Authentic-48K.csv")
    df_fake = pd.read_csv("../Fake News Dataset/LabeledFake-1K.csv")
    df_iteration(df_true, "Authentic")
    df_iteration(df_fake, "Fake")


# To create the appendix table
# appendix()

dataDistribution()

# categories()

