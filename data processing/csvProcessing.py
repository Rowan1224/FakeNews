import pandas as pd
import os


def remove_files(path, file_list):
    for f in file_list:
        file = path+"/"+f
        os.remove(file)


def fix_domain(df, col_name):

    dictionary = df[col_name].value_counts().to_dict()
    wrong_list_of_rows = []
    fixed_list_of_rows = []
    for c, count in dictionary.items():
        c2 = str(c).replace("\ufeff", "")
        # print(c2)
        wrong_list_of_rows.append(c)
        fixed_list_of_rows.append(c2)
    df = df.replace("prothomalo.com2", "prothomalo.com")
    df = df.replace("mzamin.co", "mzamin.com")
    df = df.replace("prothomalo.co", "prothomalo.com")
    df = df.replace(wrong_list_of_rows, fixed_list_of_rows)
    return df


def fix_relation(df):
    df = df.replace("related", "Related")
    df = df.replace(" related", "Related")
    df = df.replace("Related ", "Related")
    df = df.replace("related ", "Related")
    df = df.replace("unrelated", "Unrelated")
    df = df.replace("not related", "Unrelated")

    dictionary = df.relation.value_counts().to_dict()
    for c, count in dictionary.items():
        if c != "Related" and c != "Unrelated":
            print(c+" "+str(count))
            df = df[df["relation"] != c]

    return df


def fix_date(df):

    df = df.replace("018-09-19 19:37:41", "2018-09-19 19:37:41")
    df = df.replace("018-09-19 21:25:43", "2018-09-19 21:25:43")
    dictionary = df.date.value_counts().to_dict()
    wrong_list_of_rows = []
    fixed_list_of_rows = []
    for c, count in dictionary.items():
        c = c.lstrip()
        if c[0] != "2":
            l = list(c)
            l[0] = ""
            c2 = "".join(l)
            wrong_list_of_rows.append(c)
            fixed_list_of_rows.append(c2)

        # print(c)
    df = df.replace(wrong_list_of_rows, fixed_list_of_rows)
    return df


def true():
    path = "../Data/Corpus/LabeledAuthenticCorpus.csv"
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.drop_duplicates(subset="content", keep="first")

    # # Generalize the relation with dataset
    # df = fix_relation(df)
    #
    # # Fix domain
    # df = fix_domain(df, "domain")
    # dictionary = df.domain.value_counts().to_dict()
    # for c, count in dictionary.items():
    #     print(c)
    #
    # # Fix the Date Column
    # df = fix_date(df)
    # dictionary = df.date.value_counts().to_dict()
    # for c, count in dictionary.items():
    #     print(c)
    #
    # # Fix Category column
    # categories = df.category.unique()
    # for c in categories:
    #     print(c)

    categories = df.source.unique()
    print(len(categories))
    for c in categories:
        print(c)


    print(df.shape)
    df = df.dropna()
    print(df.shape)
    print(list(df))
    # df.to_csv(path, index=False)


def fake():
    fbase = "/home/rowan/PycharmProjects/FakeNews/Data/Labeled Dataset/C+S+F"
    path = "../Data/Corpus/FakeCorpusV2.csv"
    df = pd.read_csv(path)
    # mask = df.content.duplicated(keep=False)
    # duplicate = df[mask]
    # # remove_files(fbase, duplicate.articleID.unique())
    # print(duplicate.shape)
    # df = df.dropna()
    print(df.shape)
    # df.to_csv(path, index=False)


def true_48():
    path = "../Data/Corpus/AuthenticCorpus.csv"
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.drop_duplicates(subset="content", keep="first")
    print(df.shape)
    # dictionary = df.category.value_counts().to_dict()
    # wrong_list_of_rows = []
    # fixed_list_of_rows = []
    # for c, count in dictionary.items():
    #     c2 = str(c).lstrip()
    #     wrong_list_of_rows.append(c)
    #     fixed_list_of_rows.append(c2)
    #     # print(c)
    #
    # df = df.replace(wrong_list_of_rows, fixed_list_of_rows)
    #
    # categories = df.category.unique()
    # keys = new_categories()
    # count = 0
    # wrong_list_of_rows = []
    # fixed_list_of_rows = []
    # for c in categories:
    #     wrong_list_of_rows.append(c)
    #     fixed_list_of_rows.append(keys[c])
    # df = df.replace(wrong_list_of_rows, fixed_list_of_rows)
    # categories = df.category.unique()
    # for c in categories:
    #     print(c)
    # print(count)
    # print(len(categories))
    # print(df.shape)
    # df = df.dropna()
    # print(df.shape)
    # print(list(df))
    # df.to_csv(path, index=False)

    df = df.content.value_counts()
    print(df)


def new_categories():

    path = "../Data/Labeled Dataset/AllData/fix-it-felix.txt"
    keys = {}
    with open(path, 'r') as file:
        for line in file:
            line = line.replace("\n", "")
            l = len(line.split())
            key = " ".join(line.split()[:l-1])
            val = line.split()[l-1]
            # print(key)
            keys[key] = val

    print(len(keys))
    return keys


#
# fake()
true()
# true_48()
# new_categories()
