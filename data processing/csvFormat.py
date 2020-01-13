import pandas as pd 
import csv
import os
import math
from random import shuffle


dataDict = {}

def readCSV():
    data = pd.read_csv("../Data/Corpus/News-Headline.csv")
    newsList = []
    # Preview the first 5 lines of the loaded data
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    for headline, news in data.iterrows():
        article = str(news["News"]).replace("\n","")
        newsList.append(article)
        dataDict[article] = 1

        # print(news)
    return newsList

def randomize_files(file_list):
    shuffle(file_list)
    return file_list


def get_training_and_testing_sets(file_list):
    split = 0.7
    split_index = math.floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing


def textToCsv(ArtcleList,type):
    dir = "../Data/CSV"
    ID = 0

    with open(dir + '/' + type + '.csv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter=',')
        if type == "test":
            row = ["ID","Article"]
            tsv_writer.writerow(row)
        else:
            row = ["ID","Label", "Article"]
            tsv_writer.writerow(row)


        for article in ArtcleList:
            label = dataDict[article]
            if type == "train":
                row = [ID,label,article]
                tsv_writer.writerow(row)
            else:
                row = [ID,article]
                tsv_writer.writerow(row)
            ID += 1


def textTotsv(ArtcleList,type):
    dir = "../Data/TSV"
    ID = 0

    with open(dir + '/' + type + '.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        if type == "test":
            row = ["ID","Article"]
            tsv_writer.writerow(row)


        for article in ArtcleList:
            label = dataDict[article]
            if type == "train" or "dev":
                row = [ID,label,"a",article]
            else:
                row = [ID,article]
            ID += 1
            tsv_writer.writerow(row)



def readFile(newsType):
    newsList = []
    label = 0
    if newsType == 'True':
        label = 1
    else:
        label = 0
    global dataDict
    dir = "../Data/Dataset/"+newsType
    files = os.listdir(dir)
    for file in files:
        file = dir+"/"+file
        with open(file, 'r') as Infile:
            data = list(Infile)
            if len(data) < 8:
                print(file)
            else:
                line = str(data[7]).replace("\n","")
                newsList.append(line)
                dataDict[line] = label
    return newsList


def FixDataset(path):
    data = pd.read_csv(path, header=None)
    data.columns = ["Label", "Article"]
    print(data.Label.count())



TrueArticles = readCSV()
SatireArticles = readFile("Satire")
AllData = TrueArticles + SatireArticles
AllData = randomize_files(AllData)
train,test = get_training_and_testing_sets(list(AllData))
train, dev = get_training_and_testing_sets(train)
textTotsv(train, "train")
textTotsv(dev, "dev")
textTotsv(test, "test")
print(AllData)

FixDataset("/media/MyDrive/Project Fake news/paper with dataset/Hannah2017/newsfiles/fulltrain.csv")
