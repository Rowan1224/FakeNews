import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# load make_blobs to simulate data
from sklearn.datasets import make_blobs
# load decomposition to do PCA analysis with sklearn
from sklearn import decomposition

def data():
    Featurelist = []
    df = pd.read_csv("Data/Corpus/NewFeature.csv", usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    print(df.shape)
    # Iterate over each row
    for index, rows in df.iterrows():
        Featurelist.append(list(rows))

    df = pd.read_csv("Data/Corpus/NewTarget.csv")
    saved_column = df.label
    print(list(saved_column))

    X_train, X_test, y_train, y_test = train_test_split(Featurelist, list(saved_column), test_size=0.3,
                                                        random_state=109)

    return np.array(X_train), np.array(y_train)

X1, Y1 = data()
X = preprocessing.scale(X1)
print(X)


pca = decomposition.PCA(n_components=2)
pc = pca.fit_transform(X)
pc_df = pd.DataFrame(data = pc ,
        columns = ['PC1', 'PC2'])
pc_df['Cluster'] = Y1
pc_df.head()

df = pd.DataFrame({'var':pca.explained_variance_ratio_,
             'PC':['PC1','PC2']})

sns.barplot(x='PC',y="var",
           data=df, color="c")
plt.show()
fig = sns.lmplot( x="PC1", y="PC2",
  data=pc_df,
  fit_reg=False,
  hue='Cluster', # color by cluster
  legend=True,
  scatter_kws={"s": 100}) # specify the point size

plt.show()
fig.savefig('PCA.png')