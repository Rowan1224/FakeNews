import os
import csv
import pandas as pd
import requests

from bs4 import BeautifulSoup


def allreferences():
    file_dir = "/media/MyDrive/Project Fake news/Data/Model - Data/AllTrueData(Unformat)"
    news_list = os.listdir(file_dir)
    print(len(news_list))
    i = 5
    ref_set = {'ref'}
    for n in news_list:
        direct = file_dir + "/" + n
        with open(direct, "r") as file:
            j = 1
            for line in file:
                if j == i:
                    ref_set.add(line.replace("\n", ""))
                j = j + 1

    print(ref_set)

    with open("/media/MyDrive/Project Fake news/Data/Ref-class", "w") as outfile:
        for ref in ref_set:
            outfile.write(ref + "\n")


def reftrsutlevel():
    # with open('Data/Corpus/ref.csv', 'r') as readFile:
    #     reader = csv.reader(readFile)
    #     lines = list(reader)
    #     print(lines[0])
    df = pd.read_csv("Data/Corpus/ref.csv")
    df= df.drop(columns=['Timestamp'])
    print( df.mean())


def domain():
    file_dir = "Data/News"
    news_list = os.listdir(file_dir)
    print(len(news_list))
    i = 1
    domain_set = {'ref'}
    for n in news_list:
        direct = file_dir + "/" + n
        with open(direct, "r") as file:
            j = 1
            for line in file:
                if j == i:
                    # print(line)
                    line = line.replace("\ufeff","")
                    domain_set.add(line.replace("\n", ""))
                j = j + 1
    domain_set.remove("ref")
    print(domain_set)

    with open("Data/domain-list", "w") as outfile:
        for ref in domain_set:
            outfile.write(ref + "\n")


def alexaranking():
    with open("Data/domain-list", "r") as outfile:
        ranks = {}
        for d in outfile:
            d = d.replace("\n","")
            url = 'https://www.alexa.com/siteinfo/'+d
            res = requests.get(url)
            soup = BeautifulSoup(res.text, 'html.parser')
            outputs = []
            for soup in soup.find_all(class_='big data'):
                rank = str(soup.getText()).replace("\n", "")
                rank = rank.replace("\t", "")
                rank = rank.replace("#", "")
                rank = rank.replace(" ", "")
                rank = rank.replace(",", "")
                outputs.append(rank)
            if len(outputs) > 0:
                ranks[d] = outputs[0]
            else:
                ranks[d] = "0"
                print(d)
        print(ranks)
    with open('Data/Corpus/alexa-ranking.csv', 'w') as featureFile:
        writer = csv.writer(featureFile)
        for r in ranks:
            writer.writerow([r,ranks[r]])


reftrsutlevel()