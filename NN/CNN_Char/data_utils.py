import numpy as np
import pandas as pd
import re
import csv

from sklearn.model_selection import train_test_split


class Data(object):
    """
    Class to handle loading and processing of raw datasets.
    """
    def __init__(self,
                 alphabet="অআইঈউঊঋএঐওঔা ি ী ু ূ ৃ ে ৈ ো ৌকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎ ং ঃ ঁ।”“’০১২৩৪৫৬৭৮৯-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 input_size=1014, num_of_classes=2):
        """
        Initialization of a Data object.

        Args:
            data_source (str): Raw data file path
            alphabet (str): Alphabet of characters to index
            input_size (int): Size of input features
            num_of_classes (int): Number of classes in data
        """
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}  # Maps each character to an integer
        self.no_of_classes = num_of_classes
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx + 1
        self.length = input_size
        # self.data_source = data_source

    def load_data(self):
        """
        Load raw data from the source file into data variable.

        Returns: None

        """
        # data = []
        # with open(self.data_source, 'r', encoding='utf-8') as f:
        #     rdr = csv.reader(f, delimiter=',', quotechar='"')
        #     for row in rdr:
        #         txt = ""
        #         for s in row[1:]:
        #             txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
        #         data.append((int(row[0]), txt))  # format: (label, text)

        # df = pd.read_csv("../../Data/Corpus/AllDataTarget.csv")
        # data = []
        # for row in df.iterrows():
        #     news = row[1]['news']
        #     label = row[1]['label']
        #     data.append((label, news))

        df = pd.read_csv("../../Data/Corpus/AllDataTarget.csv")
        X = df["news"].values
        Y = df["label"].values
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=109)

        train_data = []
        test_data = []
        l = int(len(x_train)/8)
        for i in range(l):
            label = y_train[i]
            news = x_train[i]
            train_data.append((label, news))

        l = int(len(x_test)/4)
        test_true = []
        for i in range(l):
            label = y_test[i]
            news = x_test[i]
            test_data.append((label, news))
            test_true.append(label)
        self.train = np.array(train_data)
        self.test = np.array(test_data)
        self.test_true = np.array(test_true)
        # print("Data loaded from " + self.data_source)

    def get_all_data(self, flag):
        """
        Return all loaded data from data variable.

        Returns:
            (np.ndarray) Data transformed from raw to indexed form with associated one-hot label.

        """
        if flag == 'train':
            data = self.train
        elif flag == 'test':
            data = self.test
        else:
            return
        data_size = len(data)
        start_index = 0
        end_index = data_size
        batch_texts = data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.str_to_indexes(s))
            c = int(c) - 1
            classes.append(one_hot[c])
            # classes.append(c)
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def str_to_indexes(self, s):
        """
        Convert a string to character indexes based on character dictionary.
        
        Args:
            s (str): String to be converted to indexes

        Returns:
            str2idx (np.ndarray): Indexes of characters in s

        """
        s = s.lower()
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx
