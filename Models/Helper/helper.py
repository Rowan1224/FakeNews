
from sklearn import metrics
import os.path
import torch

def getResult(y_test, y_pred):

    if torch.is_tensor(y_test) and torch.is_tensor(y_pred):
        y_test, y_pred = cudaTocpu(y_test, y_pred)
    
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    true = report['1']
    fake = report['0']

    overall = {"Accuracy": metrics.accuracy_score(y_test, y_pred), "recall": metrics.recall_score(y_test, y_pred),
               "f1-score": metrics.f1_score(y_test, y_pred), "precision": metrics.precision_score(y_test, y_pred) }

    return true, fake, overall


def printResult(experiment, overall, fake):
    experiment = experiment.ljust(14)
    res = "{}     {:.2f}         {:.2f}        {:.2f}      #  {:.2f}         {:.2f}         {:.2f}".format(experiment,overall['precision'],overall['recall'],overall['f1-score'],fake['precision'],fake['recall'],fake['f1-score'])
    return res

def getReport(y_test,y_pred):

    if torch.is_tensor(y_test) and torch.is_tensor(y_pred):
        y_test, y_pred = cudaTocpu(y_test, y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1-Score:", metrics.f1_score(y_test, y_pred))
    print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

def saveResults(path,res):
    if os.path.isdir('./results') == False:
        os.mkdir('results')
    path = 'results/'+path
    if os.path.exists('./'+path)==False:
        with open(path, 'w', encoding="utf8") as file:
            file.write("                                Overall               #               Fake                \n")
            file.write("                   precision    recall      f1-score  #  precision    recall      f1-score\n")
            file.write(res+"\n")
    else:
        with open(path, 'a', encoding="utf8") as file:
            file.write(res+"\n")

def cudaTocpu(y_test,y_pred):
    y_test = [ y.cpu() if y.is_cuda else y for y in y_test ]
    y_pred = [ y.cpu() if y.is_cuda else y for y in y_pred ]

    return y_test, y_pred