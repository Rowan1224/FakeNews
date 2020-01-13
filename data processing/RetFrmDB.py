import csv

import mysql.connector as mariadb
import pandas as pd


import os

con = mariadb.connect(user='rowan', password='', host='localhost', database='News', unix_socket="/var/run/mysqld/mysqld.sock")
filepath = "/home/rowan/Desktop/category list"
dir = "/media/MyDrive/Project Fake news/Data/new data/"


cur = con.cursor()

i = 0

# listOfDomain = []
# for row in rows:
#     for r in row:
#         listOfDomain.append(r)
#
#for domain in listOfDomain:

# set_of_domain = set()
# with open(filepath, 'r', encoding='utf8') as file:
#         for line in file:
#             category =line.replace('\n', '')
#             if(category not in set_of_domain):
#                 set_of_domain.add(category)
#                 os.mkdir(dir+category)
#                 news = con.cursor(buffered=True)
#                 news.execute("""select  date,domain,type,headline,content from parse_news where category = %s""", (category,))
#                 data=news.fetchmany(400)
#                 for d in data:
#                     #open a file
#                     news_file = dir+category+"/"+category+"_"+str(i)+".txt"
#                     with open(news_file, 'w', encoding='utf8') as outfile:
#                         i = i+1
#                         for r in d:
#                             #write data here by a new line on every iteration
#                             outfile.write(str(r)+"\n")
#                             print(r)
#
#                 news.close()
#
# #print(rows)
# print(i)
# cur.close()
path = "../Data/Unformat/"


def writetofile(ID, domain, date, type, headline, content):
    filename = path+ID
    with open(filename, "w") as infile:
        infile.write(domain+"\n")
        infile.write(str(date) + "\n")
        infile.write(type + "\n")
        infile.write(headline + "\n")
        infile.write(content)
        infile.close()


def ret_news_headline():
    news = con.cursor(buffered=True)
    news.execute("""select domain,date,category,headline,content from parse_news""")

    ID = 0
    print(news.rowcount)
    with open('../Data/Corpus/AuthenticCorpus.csv', 'w') as CSV:
        writer = csv.writer(CSV)
        headers = ["articleID", "domain", "date", "category", "relation", "headline", "content", "label"]
        writer.writerow(headers)
        for domain, date, category, headline, content in news:
            ID += 1
            row = [str(format(ID, '04d')), domain, str(date), category, "related", headline, content, 1]
            writer.writerow(row)


df = pd.read_csv("../Data/Corpus/FakeCorpus.csv")
print(df.shape)
df = df.drop_duplicates(subset="content", keep="first")
print(df.shape)
# # headline = df["headline"].values
# # headline = set(headline)
# # print(len(headline))
ret_news_headline()
df = pd.read_csv("../Data/Corpus/AuthenticCorpus.csv")
print(df.shape)
df = df.drop_duplicates(subset="content", keep="first")
df.to_csv("../Data/Corpus/AuthenticCorpus.csv")
print(df.shape)

# for row in df.iterrows():
#     articleID, domain, date, type, headline, content = row[1]
#     writetofile(articleID, domain, date, type, headline, content)

