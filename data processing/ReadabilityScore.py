import unicodedata
import numpy as np

def countChars(wordList):
    count = 0
    for word in wordList:
        l = list(map(unicodedata.name,word))
        count += len(l)

    return count


import re, string, collections, os, random
import csv
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
char_set = set()
articleID = 0
stopWords = set()

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

    with open('../Data/Corpus/word-freqlist.csv', 'r') as f:
        reader = csv.reader(f) # pass the file to our csv reader
        for row in reader:     # iterate over the rows in the file
            word_freq_Dict[row[0]] = row[1]

    with open('../Data/Corpus/alexa-ranking.csv', 'r') as f:
        reader = csv.reader(f) # pass the file to our csv reader
        for row in reader:     # iterate over the rows in the file
            alexa_rank_dict[row[0]] = row[1]
    with open('../Data/Corpus/StopWords', 'r') as f:
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
    word_set = set()
    word_list = str(line).split(" ")
    totalWords = len(word_list)
    totalChars = countChars(word_list)
    for word in word_list:
        word = RemPunctuation(word)
        word_set.add(word)


    return totalWords,totalChars




def textanalysis(file_path):

    article = ""

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


    wordLengthArticle , charCount= wordcounter(article)

    totalsentenceInArticle = puncutaionCounter(article)[2]

    ColemanLiauIndex = 5.89 * (charCount/wordLengthArticle) - 0.3 * (totalsentenceInArticle/wordLengthArticle) - 15.8

    AutomatedReadabilityIndex = 4.71 * (charCount / wordLengthArticle) + 0.5 *(wordLengthArticle / totalsentenceInArticle) - 21.43

    if wordLengthArticle < 100:
        score2 = 0

    return score2


loadCorpus()
dataDirectory = "../Data/News2"
articlelist = os.listdir(dataDirectory)
random.shuffle(articlelist)
trueScorelist = []
fakeScorelist = []

for article in articlelist:


    articleID = articleID + 1
    label = 0
    if str(article.split("-")[0]) == "True":
        label = 1
    score = textanalysis(dataDirectory+"/"+article)
    if label == 0:
        if score > 0:
            fakeScorelist.append(score)
    else:
        if score > 0:
            trueScorelist.append(score)




trueAvgScore = np.average(np.asarray(trueScorelist))
fakeAvgScore = np.average(np.asarray(fakeScorelist))

print("True "+str(trueAvgScore))
print("Fake "+str(fakeAvgScore))



# for char in char_set:
#     if string.punctuation.__contains__(char):
#         print(char)
#     #print(char.encode('raw_unicode_escape'))

