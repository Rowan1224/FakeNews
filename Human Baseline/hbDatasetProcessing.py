import pandas as pd


column = ["articleID", "label", "news"]
data = []
ID = 0
with open("../Data/hb.txt", "r") as file:
    for line in file:
        ID += 1
        label = 0
        l = line.split(" ")[0]
        if l == "true":
            label = 1
        news = line.replace(l+" ", "")
        news = news.replace("\n", "")
        row = [ID, label, news]
        data.append(row)

df = pd.DataFrame(data, columns=column)
df.to_csv("../Data/HB.csv")
print(df.head())

all = pd.read_csv("../Data/Corpus/AllDataTarget.csv")
print(all.shape)
print(df.shape)
intersected_df = pd.merge(all[['news','label']], df[['news', 'label']], how='left', on=["news", "label"], indicator= True)[lambda x: x._merge=='left_only'].drop('_merge',1)
# intersected_df = all.merge(df, indicator=True, how="left", on=)[lambda x: x._merge=='left_only'].drop('_merge',1)

print(intersected_df.shape)

# for idx, row in df.iterrows():
#     print(row['news'])
# print(all["news"].str.find("সংসদ"))