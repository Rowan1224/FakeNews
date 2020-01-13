import csv
import os
import pandas as pd


# fbase = "/home/rowan/PycharmProjects/FakeNews/Data/Labeled Dataset/C+S+F"
v1base = "/home/rowan/PycharmProjects/FakeNews/Data/Labeled Dataset/Labeled Satire"

v1 = os.listdir(v1base)


def fix_relation(df):
    df = df.replace("Misc", "Miscellaneous")
    df = df.replace("miscellaneous", "Miscellaneous")
    df = df.replace("international ", "International")
    df = df.replace("NAtional ", "National")
    df = df.replace("politics", "Politics")
    df = df.replace("lifestyle", "Lifestyle")
    df = df.replace("LIfestyle", "Lifestyle")
    df = df.replace("NAtional ", "National")
    df = df.replace("national", "National")
    df = df.replace("crime", "Crime")
    df = df.replace("international", "International")
    df = df.replace("not Related", "Unrelated")
    df = df.replace("Not Related", "Unrelated")
    df = df.replace("http://www.earki.com", "earki.com")
    df = df.replace("www.aparadhchokh24bd.com", "aparadhchokh24bd.com")
    df = df.replace("0:00", "2019-03-14T02:33:32+00:00")
    df = df.replace("related ", "Related")
    df = df.replace("unrelated", "Unrelated")
    df = df.replace("not related", "Unrelated")

    return df


def fix_domain(df, domain):
    for d in domain:

        nd = d.replace("www.", "")
        df = df.replace(d, nd)

    return df


def doc_to_list_fake(path, articleID, label, f_type):
    data = [articleID]
    temp =[]
    with open(path, "r") as infile:
        for line in infile:
            line = line.replace("\n", "")
            if line == "":
                continue
            temp.append(line)
    if len(temp) == 8:

        for i in range(8):
            # if i == 0:
            #     data.append(temp[i].replace(" ", ""))
            if i == 2:
                continue
            elif i == 1:
                # if temp[i] == '0:00':
                #     print("v1 3 " + path.replace(v1base, ""))
                d = temp[i].lstrip()
                data.append(d)
            else:
                if not temp[i]:
                    print("v1 8 " + path.replace(v1base, ""))
                d = temp[i].lstrip()
                data.append(d)

        data.append(label)
        data.append(f_type)
        if len(data) != 10:
            print("v1 code " + path.replace(v1base, ""))

        return data
    else:
        print(path)
        return [articleID]


def text_to_csv_fake():

    fake_news_list = os.listdir(v1base)

    with open('../Data/Corpus/FakeCorpusV3.csv', 'w') as CSV:
        writer = csv.writer(CSV)
        headers = ["articleID", "domain", "date", "category", "source", "relation", "headline", "content", "label", "F-type"]
        writer.writerow(headers)
        articleID = 0
        for doc in fake_news_list:
            path = v1base + "/" + doc
            articleID += 1
            label = 0
            f_type = doc.split("-")[0]
            writer.writerow(doc_to_list_fake(path, str(format(articleID, '04d')), label, f_type))


# text_to_csv_fake()
path = "../Data/Corpus/FakeCorpusV4.csv"
df = pd.read_csv(path)
print(list(df))
# df = df.drop(['source', 'relation', 'F-type'], axis=1)
# print(list(df))
# path = "../Data/Corpus/FakeCorpusV4.csv"
# df.to_csv(path, index=False)
# df = df.dropna()
# df = df.drop_duplicates(subset="content", keep="first")
# df = fix_relation(df)
# domain = df['domain'].unique()
# df = fix_domain(df, domain)
# print(df.shape)
# categories = df['domain'].unique()
# for c in categories:
#     print(c)
# df.to_csv(path, index=False)