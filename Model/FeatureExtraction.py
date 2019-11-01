import re, string, collections, os, random
import csv
import pandas as pd
import unicodedata

puncList = []
word_freq_Dict = {}
alexa_rank_dict = {}
char_set = set()
articleID = 0
stopWords = set()

def punctuationLoader():
    puncList.append("।")
    puncList.append("”")
    puncList.append("“")
    puncList.append("’")

    for p in string.punctuation.lstrip():
        puncList.append(p)


punctuationLoader()


def RemPunctuation(word):

    for p in puncList:
        word = word.replace(p, "")

    return word

def loadCorpus():

    with open('Data/Corpus/word-freqlist.csv', 'r') as f:
        reader = csv.reader(f) # pass the file to our csv reader
        for row in reader:     # iterate over the rows in the file
            word_freq_Dict[row[0]] = row[1]

    with open('Data/Corpus/alexa-ranking.csv', 'r') as f:
        reader = csv.reader(f) # pass the file to our csv reader
        for row in reader:     # iterate over the rows in the file
            alexa_rank_dict[row[0]] = row[1]
    with open('Data/Corpus/StopWords', 'r') as f:
        for row in f:     # iterate over the rows in the file
            row = row.replace("\n", "")
            stopWords.add(row)

def countChars(wordList):
    count = 0
    for word in wordList:
        word = RemPunctuation(word)
        try:
            l = list(map(unicodedata.name,word))
            count += len(l)
        except:
            print(word)


    return count

def puncutaionCounter(line):
    totalPunctuation = 0
    puncCountDict = {}
    numberOfSentence = 0
    char_list = str(line).lstrip()
    for char in char_list:
       if char in puncList:
           totalPunctuation = totalPunctuation + 1
           if char in puncCountDict.keys():
               puncCountDict[char] = puncCountDict[char] + 1
           else:
               puncCountDict[char] = 1
    eospunctuation = ["!", "।", "?"]

    for p in eospunctuation:
        if p in puncCountDict.keys():
            numberOfSentence = puncCountDict[p] + numberOfSentence

    if numberOfSentence == 0:
        # print(line)
        numberOfSentence = 1

    return totalPunctuation, puncCountDict, numberOfSentence


def wordcounter(line):
    word_set = set()
    word_list = str(line).split(" ")
    totalWords = len(word_list)
    totalChars = countChars(word_list)
    for word in word_list:
        word = RemPunctuation(word)
        word_set.add(word)


    return totalWords,word_set,totalChars


def wordfrequency(doc):

    mostFrequent = 0
    leastFrequent = 0
    for word in doc:
        word = RemPunctuation(word)
        if word in word_freq_Dict:
            mostFrequent = mostFrequent + 1
        else:
            leastFrequent = leastFrequent + 1

    return mostFrequent, leastFrequent

def alexaRanking(line):
    line = line.replace("\ufeff", "")
    line = line.replace("\n", "")
    if line in alexa_rank_dict:
        return (int(alexa_rank_dict[line])/100000)
    else:
        return (122000/100000)

def stopwordsfreq(line):
    totalwords = 0
    word_list = str(line).split(" ")
    for word in word_list:
        word = RemPunctuation(word)
        if word in stopWords:
            totalwords = totalwords + 1

    return totalwords




def textanalysis(file_path):
    headline = ""
    article = ""
    domain = ""
    with open(file_path,"r") as file:
        i = 0
        for line in file:
            line = line.replace("\n","")
            i = i + 1
            if i == 1:
                domain = line
            if i < 7:
                continue
            if i == 7:
                headline = line
                continue
            article = article + line

    wordLengthHeadline = wordcounter(headline)[0]
    wordDictHeadline = wordcounter(headline)[1]
    totalPunctuationInHeadline = puncutaionCounter(headline)[0]
    puncDictHeadline = puncutaionCounter(headline)[1]
    totalsentenceInHeadline = puncutaionCounter(headline)[2]
    numberOfMostFrequentWordHeadline = wordfrequency(wordDictHeadline)[0]
    numberOfLeastFrequentWordHeadline = wordfrequency(wordDictHeadline)[1]


    wordLengthArticle,wordDictArticle,charCount = wordcounter(article)
    totalPunctuationInArticle,puncDictArticle,totalsentenceInArticle = puncutaionCounter(article)

    numberOfMostFrequentWordArticle, numberOfLeastFrequentWordArticle = wordfrequency(wordDictArticle)
    # numberOfLeastFrequentWordArticle = wordfrequency(wordDictArticle)[1]
    numberOfStopWordArticle = stopwordsfreq(article)


    ColemanLiauIndex = 5.89 * (charCount / wordLengthArticle) - 0.3 * (
                totalsentenceInArticle / wordLengthArticle) - 15.8

    AutomatedReadabilityIndex = 4.71 * (charCount / wordLengthArticle) + 0.5 * (
                wordLengthArticle / totalsentenceInArticle) - 21.43

    featureDict ={}

    featureDict["nwh"] = wordLengthHeadline
    featureDict["nwa"] = wordLengthArticle
    featureDict["nsh"] = totalsentenceInHeadline
    featureDict["nsa"] = totalsentenceInArticle
    featureDict["avg_wps"] = wordLengthArticle / totalsentenceInArticle
    featureDict["nph"] = totalPunctuationInHeadline
    featureDict["npa"] = totalPunctuationInArticle
    if "!" in puncDictHeadline.keys():
        featureDict["nesh"] = puncDictHeadline["!"]
    else:
        featureDict["nesh"] = 0

    if "!" in puncDictArticle.keys():
            featureDict["nesa"] = puncDictArticle["!"]
    else:
            featureDict["nesa"] = 0

    featureDict["mfwa"] = numberOfMostFrequentWordArticle
    featureDict ["lfwa"] = numberOfLeastFrequentWordArticle
    featureDict["rank"] = alexaRanking(domain)
    featureDict["nsw"] = numberOfStopWordArticle
    featureDict["CLI"] = ColemanLiauIndex
    featureDict["ARI"] = AutomatedReadabilityIndex
    featureDict["news"] = article
    featureDict["headline"] = headline

    # print(featureDict)
    return featureDict


def textanalysisFromCSV(domain,headline, article):

    article = str(article).replace("\n","")
    wordLengthHeadline = wordcounter(headline)[0]
    wordDictHeadline = wordcounter(headline)[1]
    totalPunctuationInHeadline = puncutaionCounter(headline)[0]
    puncDictHeadline = puncutaionCounter(headline)[1]
    totalsentenceInHeadline = puncutaionCounter(headline)[2]
    numberOfMostFrequentWordHeadline = wordfrequency(wordDictHeadline)[0]
    numberOfLeastFrequentWordHeadline = wordfrequency(wordDictHeadline)[1]


    wordLengthArticle,wordDictArticle,charCount = wordcounter(article)
    totalPunctuationInArticle,puncDictArticle,totalsentenceInArticle = puncutaionCounter(article)
    numberOfMostFrequentWordArticle, numberOfLeastFrequentWordArticle = wordfrequency(wordDictArticle)
    # numberOfLeastFrequentWordArticle = wordfrequency(wordDictArticle)[1]
    numberOfStopWordArticle = stopwordsfreq(article)


    ColemanLiauIndex = 5.89 * (charCount / wordLengthArticle) - 0.3 * (
                totalsentenceInArticle / wordLengthArticle) - 15.8

    AutomatedReadabilityIndex = 4.71 * (charCount / wordLengthArticle) + 0.5 * (
                wordLengthArticle / totalsentenceInArticle) - 21.43

    featureDict ={}

    featureDict["nwh"] = wordLengthHeadline
    featureDict["nwa"] = wordLengthArticle
    featureDict["nsh"] = totalsentenceInHeadline
    featureDict["nsa"] = totalsentenceInArticle
    featureDict["avg_wps"] = wordLengthArticle / totalsentenceInArticle
    featureDict["nph"] = totalPunctuationInHeadline
    featureDict["npa"] = totalPunctuationInArticle
    if "!" in puncDictHeadline.keys():
        featureDict["nesh"] = puncDictHeadline["!"]
    else:
        featureDict["nesh"] = 0

    if "!" in puncDictArticle.keys():
            featureDict["nesa"] = puncDictArticle["!"]
    else:
            featureDict["nesa"] = 0

    featureDict["mfwa"] = numberOfMostFrequentWordArticle
    featureDict ["lfwa"] = numberOfLeastFrequentWordArticle
    featureDict["rank"] = alexaRanking(domain)
    featureDict["nsw"] = numberOfStopWordArticle
    featureDict["CLI"] = ColemanLiauIndex
    featureDict["ARI"] = AutomatedReadabilityIndex
    featureDict["news"] = article
    featureDict["headline"] = headline

    # print(featureDict)
    return featureDict


loadCorpus()
dataDirectory = "Data/News"
articlelist = os.listdir(dataDirectory)
random.shuffle(articlelist)
csvheaderlist = ["articleID","nwh","nwa","nsh","nsa","avg_wps","nph","npa","nesh","nesa","mfwa","lfwa","rank","nsw","CLI","ARI","label"]
df = pd.read_csv("Data/Corpus/AllTrueNews.csv")

with open('Data/Corpus/AllDataFeature.csv', 'w') as featureFile, open('Data/Corpus/AllDataTarget.csv', 'w') as targetFile:
    writer = csv.writer(featureFile)
    target = csv.writer(targetFile)
    writer.writerow(csvheaderlist)

    target.writerow(["articleID","label","news"])
    for article in articlelist:

        # print(article.split("-")[0])
        articleID = articleID + 1
        label = 0
        if str(article.split("-")[0]) == "True":
            label = 1

        if label == 1:
            continue

        featurelist = []
        featureDict = textanalysis(dataDirectory+"/"+article)
        featureDict["label"] = label
        featureDict["articleID"] = articleID


        flag = True
        for feature in csvheaderlist:
            if featureDict[feature] < 0:
                flag = False
                continue
            featurelist.append(featureDict[feature])

        # print(featuelist)
        if flag:
            target.writerow([str(format(articleID, '04d')), label, featureDict["news"]])
            writer.writerow(featurelist)

    for row in df.iterrows():
        articleID = articleID + 1
        label = 1
        # target.writerow([str(format(articleID, '04d')), label])
        domain = row[1]["Domain"]
        headline = row[1]["Headline"]
        news = row[1]["News"]

        featurelist = []
        featureDict = textanalysisFromCSV(domain,headline,news)
        featureDict["label"] = label
        featureDict["articleID"] = articleID

        flag = True
        for feature in csvheaderlist:
            featurelist.append(featureDict[feature])

            if featureDict[feature] < 0:
                flag = False
                continue

        if len(featurelist) < len(csvheaderlist):
            print(row)
        if flag:
            target.writerow([str(format(articleID, '04d')), label, featureDict["news"]])
            writer.writerow(featurelist)










# for char in char_set:
#     if string.punctuation.__contains__(char):
#         print(char)
#     #print(char.encode('raw_unicode_escape'))

