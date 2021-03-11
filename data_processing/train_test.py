import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

def data_laod():
    df_true = pd.read_csv("../Fake News Dataset/Authentic-48K.csv")
    df_fake = pd.read_csv("../Fake News Dataset/Fake-1K.csv")
    df = pd.concat([df_fake, df_true])
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)
    try:
        os.mkdir("../Fake News Dataset/Train_Test/")
        df.to_csv("../Fake News Dataset/Train_Test/TrainTest.csv")
        
    except OSError as error:
        print(error)
    


def bert_train():
    df = pd.read_csv("../Fake News Dataset/Train_Test/TrainTest.csv")
    df = df[["articleID", "content", "label"]]
    train, test = train_test_split(df, test_size=0.3)

    try:
        os.mkdir("../Models/BERT/Data/")
        train.to_csv("../Models/BERT/Data/Train.csv")
        test.to_csv("../Models/BERT/Data/Test.csv")
    except OSError as error:
        print(error)



data_laod()
bert_train()