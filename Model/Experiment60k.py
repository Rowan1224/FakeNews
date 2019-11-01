import re, string, collections, os, random
import csv
import pandas as pd
# from textstat.textstat import textstat
#
# test_data = (
#     "Playing games has always been thought to be important to "
#     "the development of well-balanced and creative children; "
#     "however, what part, if any, they should play in the lives "
#     "of adults has never been researched that deeply. I believe "
#     "that playing games is every bit as important for adults "
#     "as for children. Not only is taking time out to play games "
#     "with our children and other adults valuable to building "
#     "interpersonal relationships but is also a wonderful way "
#     "to release built up tension."
# )
# textstat.set_language('')
# print(textstat.flesch_reading_ease(test_data))
# print(textstat.smog_index(test_data))
# print(textstat.flesch_kincaid_grade(test_data))
# print(textstat.coleman_liau_index(test_data))
# print(textstat.automated_readability_index(test_data))
# print(textstat.dale_chall_readability_score(test_data))
# print(textstat.difficult_words(test_data))
# print(textstat.linsear_write_formula(test_data))
# print(textstat.gunning_fog(test_data))
# print(textstat.text_standard(test_data)


puncList = []
word_freq_Dict = {}
alexa_rank_dict = {}
char_set = {"chars"}
articleID = 0
stopWords = {"stop"}

def punctuationLoader():
    puncList.append("।")
    puncList.append("”")
    puncList.append("“")
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
    word_set = {"word"}
    word_list = str(line).split(" ")
    totalWords = len(word_list)
    for word in word_list:
        word = RemPunctuation(word)
        word_set.add(word)

    word_set.remove("word")
    return totalWords,word_set


def wordfrequency(doc):

    mostFrequent = 0
    leastFrequent = 0
    for word in doc:
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
        if word in stopWords:
            totalwords = totalwords + 1

    return totalwords




def textanalysis(headline,article):



    wordLengthHeadline = wordcounter(headline)[0]
    wordDictHeadline = wordcounter(headline)[1]
    totalPunctuationInHeadline = puncutaionCounter(headline)[0]
    puncDictHeadline = puncutaionCounter(headline)[1]
    totalsentenceInHeadline = puncutaionCounter(headline)[2]
    numberOfMostFrequentWordHeadline = wordfrequency(wordDictHeadline)[0]
    numberOfLeastFrequentWordHeadline = wordfrequency(wordDictHeadline)[1]

    wordLengthArticle= wordcounter(article)[0]
    wordDictArticle = wordcounter(article)[1]
    totalPunctuationInArticle = puncutaionCounter(article)[0]
    puncDictArticle = puncutaionCounter(article)[1]
    totalsentenceInArticle = puncutaionCounter(article)[2]
    numberOfMostFrequentWordArticle = wordfrequency(wordDictArticle)[0]
    numberOfLeastFrequentWordArticle = wordfrequency(wordDictArticle)[1]
    numberOfStopWordArticle = stopwordsfreq(article)

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
    #featureDict["rank"] = alexaRanking(domain)
    featureDict["nsw"] = numberOfStopWordArticle

    print(featureDict)
    return featureDict


loadCorpus()
df = pd.read_csv("Data/Corpus/News-Headline.csv")
corpus = []
for index, rows in df.iterrows():
    corpus.append(list(rows))
csvheaderlist = ["articleID","nwh","nwa","nsh","nsa","avg_wps","nph","npa","nesh","nesa","mfwa","lfwa","nsw"]
with open('Data/Corpus/ExpFeature.csv', 'w') as featureFile:
    writer = csv.writer(featureFile)
    writer.writerow(csvheaderlist)
    for article in corpus:


        articleID = articleID + 1
        label = 0


        featuelist = [str(format(articleID, '04d'))]
        featureDict = textanalysis(article[0],article[1])
        for feature in csvheaderlist:
            if feature == "articleID":
                continue
            featuelist.append(featureDict[feature])
        print(featuelist)
        writer.writerow(featuelist)





















# for char in char_set:
#     if string.punctuation.__contains__(char):
#         print(char)
#     #print(char.encode('raw_unicode_escape'))

