import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import NN.load_data as load_data
from NN import CNN, RCNN, RNN, LSTM, selfAttention,GRU
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn import metrics
from NN import LSTM_Attn

vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()


def printScore(y_test, y_pred):
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))

    print("F1-Score:", metrics.f1_score(y_test, y_pred))

    print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))

    print(metrics.classification_report(y_test, y_pred))


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    if torch.cuda.is_available():
        model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        # print(batch)
        text, target = batch
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * (num_corrects/len(target))
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    y_pred = []
    y_test = []
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text, target = batch
            if (text.size()[0] is not 32):
                continue
            # target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            y_test.extend(target.data)
            y_pred.extend(torch.max(prediction, 1)[1].view(target.size()).data)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(target)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter), y_test, y_pred
	

learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300

epochs = 100

in_channels = 1
out_channels = 256
kernel_heights = [1, 2, 3, 4]
stride = 1
padding = 0
keep_probab = 0.8

# model = RNN.RNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model = CNN.CNN(batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, word_embeddings)
loss_fn = F.cross_entropy

for epoch in range(epochs):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    val_loss, val_acc, y_test, y_pred = eval_model(model, valid_iter)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    
test_loss, test_acc, y_test, y_pred = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
printScore(y_test, y_pred)

