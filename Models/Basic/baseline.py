import pandas as pd
from sklearn.model_selection import train_test_split
import re, string, collections, os, random
import sys
sys.path.insert(1, '../Helper/')
import helper



df = pd.read_csv("../../Fake News Dataset/Train_Test/TrainTest.csv")
labels = []
for row in df.iterrows():
    row = row[1]
    label = row["label"]
    labels.append(label)


y_train, y_test = train_test_split(labels, test_size=0.3, random_state=109)
true_df = []
fake_df = []
overall_df = []

def random_baseline():
    y_random = []

    for i in range(len(y_test)):
        b = random.choice([0, 1])
        y_random.append(b)
    t,f,o = helper.getResult(y_test, y_random)
    true_df.append(t)
    fake_df.append(f)
    overall_df.append(o)




y_major = []


for i in range(len(y_test)):
    a = 1
    b = random.choice([0, 1])
    y_major.append(a)


for i in range(10):
    random_baseline()



print("Major Baseline")
t,f,o = helper.getResult(y_test, y_major)



print("Fake")
df_result = pd.DataFrame(fake_df)
mean = df_result.mean(axis=0)
random_fake=mean.to_dict()

print("overall")
df_result = pd.DataFrame(overall_df)
mean = df_result.mean(axis=0)
random_over=mean.to_dict()

print("                                Overall               #               Fake                ")
print("                   precision    recall      f1-score  #  precision    recall      f1-score")
helper.printResult("Majority",o,f)
helper.printResult("Random",random_over,random_fake)