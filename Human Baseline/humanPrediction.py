import pandas as pd
from sklearn import metrics


def printScore(y_true, y_pred):
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_true, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_true, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_true, y_pred))

    print("F1-Score:", metrics.f1_score(y_true, y_pred))

    print("Confusion Matrix:", metrics.confusion_matrix(y_true, y_pred))

    print(metrics.classification_report(y_true, y_pred))


def confusion_matrix(y_pred, matrix):

    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if len(matrix) == i:
                d = {"True": 1, "Fake": 0}
                matrix.append(d)
            elif "True" not in matrix[i].keys():
                matrix[i]["True"] = 1
            elif "True" in matrix[i].keys():
                matrix[i]["True"] += 1

        else:
            if len(matrix) == i:
                d = {"True": 0, "Fake": 1}
                matrix.append(d)
            elif "Fake" not in matrix[i].keys():
                matrix[i]["Fake"] = 1
            elif "Fake" in matrix[i].keys():
                matrix[i]["Fake"] += 1

    return matrix


def fleiss_kappa(matrix):
    p_col = [0, 0]
    test = []
    p_row = 0
    for d in matrix:
        t = d["True"]
        f = d["Fake"]
        p_col[0] += t
        p_col[1] += f
        p = (t*t + f*f - (t+f))/((t+f)*(t+f-1))
        test.append(p)
        p_row += p

    p_mean = p_row/len(matrix)
    p1 = p_col[0]/(len(matrix)*5)
    p2 = p_col[1] / (len(matrix) * 5)
    pe_mean = ((p1*p1)+(p2*p2))
    k = (p_mean-pe_mean)/(1-pe_mean)
    print(test)
    return k

def calculate_percentage(df):
    df = df.drop([0.0], axis=1)
    df = df.fillna(0)
    df["sum"] = df.sum(axis=1)
    print(df)
    df_new = df.loc[ :].div(df["sum"], axis=0)
    print(df_new.mean())


fakeNewsID = [11, 15, 19, 22, 27, 29, 36, 38, 39, 40, 41, 43, 45, 46, 48, 49, 56, 58, 65, 66, 74, 77, 84, 91, 92, 98, 99, 101, 103, 104, 107, 108, 112, 113, 115, 118, 119, 120, 124, 125, 126, 128, 129, 131, 132, 134, 136, 137, 138, 139, 140, 141, 142, 144, 146, 147, 148, 149, 150, 151]
y_true = []
for i in range(2, 152):
   if i in fakeNewsID:
       y_true.append(0)
   else:
       y_true.append(1)


filenames = ["NN.csv", "NS.csv", "ZH.csv", "NE.csv","NEN.csv"]
matrix = []
all_predictions = []
parcentange = []
ans_true = []
ans_fake = []
with open("results.txt", "w") as outfile:
    for file in filenames:
        df = pd.read_csv(file)
        df.columns = ["News", "Q1", "Q2"]
        df = df.fillna(0)
        y_pred = []
        for row in df.iterrows():
            p = row[1].News
            if p == 1.0:
                y_pred.append(0)
            else:
                y_pred.append(1)
        matrix = confusion_matrix(y_pred, matrix)
        all_predictions.append(y_pred)
        true = df.Q2.value_counts().to_dict()
        fake = df.Q1.value_counts().to_dict()
        ans_true.append(true)
        ans_fake.append(fake)
        outfile.write("Name: "+file.replace(".csv", "")+"\n")
        outfile.write("Accuracy: "+str(metrics.accuracy_score(y_true, y_pred))+"\n")
        outfile.write("Precision: "+str(metrics.precision_score(y_true, y_pred))+"\n")
        outfile.write("Recall:"+str(metrics.recall_score(y_true, y_pred))+"\n")
        outfile.write("F1-Score: "+str(metrics.f1_score(y_true, y_pred))+"\n")
        outfile.write("Confusion Matrix: "+str(metrics.confusion_matrix(y_true, y_pred))+"\n")
        outfile.write(metrics.classification_report(y_true, y_pred)+"\n")
        outfile.write("True \n\n")

        if 1.0 in true.keys():
            outfile.write("Option 1: "+str(true[1.0])+"\n")
        if 2.0 in true.keys():
            outfile.write("Option 2: " + str(true[2.0])+"\n")
        if 3.0 in true.keys():
            outfile.write("Option 3: " + str(true[3.0])+"\n")
        if 4.0 in true.keys():
            outfile.write("Option 4: " + str(true[4.0])+"\n\n")
        outfile.write("Fake \n\n")
        if 1.0 in fake.keys():
            outfile.write("Option 1: " + str(fake[1.0])+"\n")
        if 2.0 in fake.keys():
            outfile.write("Option 2: " + str(fake[2.0])+"\n")
        if 3.0 in fake.keys():
            outfile.write("Option 3: " + str(fake[3.0])+"\n")
        if 4.0 in fake.keys():
            outfile.write("Option 4: " + str(fake[4.0])+"\n")
        outfile.write("\n###################################################################\n\n")

print(matrix)
print(fleiss_kappa(matrix))
df = pd.DataFrame(ans_fake)
print(calculate_percentage(df))
df = pd.DataFrame(ans_true)
print(calculate_percentage(df))

mean_kappa = 0

for i in range(len(all_predictions)):
    for j in range(i+1, len(all_predictions)):
        print(str(i)+" "+str(j))
        k = metrics.cohen_kappa_score(all_predictions[i], all_predictions[j])
        mean_kappa += k


print(mean_kappa/10)
