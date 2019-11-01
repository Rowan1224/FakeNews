import os
import pandas as pd
import re
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from scipy import sparse,hstack, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
# s = s.lower()
# s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
# tokens = [token for token in s.split(" ") if token != ""]
# output = list(ngrams(tokens, 5))
import string

puncList = ["।", "”", "“", "’"]
for p in string.punctuation.lstrip():
    puncList.append(p)

def tokenizer(doc):
    # remove punctuation
    tokens = []
    def cleanword(word):
        for p in puncList:
            word = word.replace(p, "")
        word = re.sub(r'[\u09E6-\u09EF]', "", word, re.DEBUG)  # replace digits

        return word

    for word in doc.split(" "):
        word = cleanword(word)
        if word != "":
            tokens.append(word)

    return tokens


dataDirectory = "../Data/News"
articlelist = os.listdir(dataDirectory)

df = pd.read_csv("../Data/Corpus/AllDataTarget.csv")

head = list(df)
# print(df.groupby("label").count())
stopWords = []
with open('../Data/Corpus/StopWords', 'r') as f:
    for row in f:  # iterate over the rows in the file
        row = row.replace("\n", "")
        stopWords.append(row)

# print(stopWords)

tfidf_char = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(3, 5), stop_words=stopWords,
                             decode_error='replace', encoding='utf-8',analyzer='char')
tfidf_word = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 3),
                             stop_words=stopWords, decode_error='replace',
                             encoding='utf-8', analyzer='word', tokenizer=tokenizer)
X = df.news



# count_vect = CountVectorizer()
# print(X_train.values.astype('U'))
# X_train_counts = count_vect.fit_transform(X_train.values.astype('U'))
X_train_tfidf_char = tfidf_char.fit_transform(X.values.astype('U'))
X_train_tfidf_word = tfidf_word.fit_transform(X.values.astype('U'))
# labels = df.label
# selector = SelectKBest(f_regression, k=10)
# k_best_features = selector.fit_transform(X_train_tfidf,Y)
# print(features.shape)
# print(X_train_tfidf)

# print(type(X_train_tfidf_char))
# df1 = pd.read_csv("../Data/Corpus/AllDataFeature.csv")
# df1 = df1[["nesa", "rank"]]
# X2 = sparse.csr.csr_matrix(df1.values)
# print(X_train_tfidf.shape)
# print(X2.shape)
X = sparse.hstack((X_train_tfidf_word, X_train_tfidf_char))
Y = df[["label"]]

print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y.values.ravel(), test_size=0.3, random_state=109)

#Create a LR Classifier
clf = svm.SVC(kernel='linear', C=10, cache_size=7000)

# clf = LogisticRegression()

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)



# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))

print("F1-Score:", metrics.f1_score(y_test, y_pred))

print("Confusion Matrix:", metrics.confusion_matrix(y_test,y_pred))


print(metrics.classification_report(y_test,y_pred))

