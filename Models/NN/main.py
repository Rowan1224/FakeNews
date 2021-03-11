import os
import time
import load_data
import CNN, LSTM_Attn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from sklearn import metrics
import argparse
   
import sys
sys.path.insert(1, '../Helper/')
import helper, config

checkpoint_history = []
early_stop_monitor_vals = []
best_score = 0


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

    
def train_model(model, train_iter, epoch, loss_fn):
    total_epoch_loss = 0
    total_epoch_acc = 0
    if torch.cuda.is_available():
        model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
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


def eval_model(model, val_iter, loss_fn):
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



def checkpoint_model(model_to_save, path_to_save, current_score, epoch, model_name, mode='max'):
    """
    Checkpoints models state after each epoch.
    :param model_to_save:
    :param optimizer_to_save:
    :param path_to_save:
    :param current_score:
    :param epoch:
    :param n_epoch:
    :param mode:
    :return:
    """


    model_state = {'epoch': epoch + 1,
                   'model_state': model_to_save.state_dict(),
                   'score': current_score,
                   }

    # Save the model as a regular checkpoint
    # torch.save(model_state, path_to_save + 'last.pth'.format(epoch))

    checkpoint_history.append(current_score)
    is_best = False

    # If the model is best so far according to the score, save as the best model state
    if ((np.max(checkpoint_history) == current_score and mode == 'max') or
            (np.min(checkpoint_history) == current_score and mode == 'min')):
        is_best = True
        best_score = current_score
        torch.save(model_state, path_to_save + '{}_best.pth'.format(model_name))

    print('Current best', max(checkpoint_history), 'after epoch {}'.format(epoch))

    return is_best


def load_saved_model(model, path):
    """
    Load a saved model from dump
    :return:
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])








def run_model(model_name):

    
    vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    output_size = config.output_size
    hidden_size = config.hidden_size
    embedding_length = config.embedding_length

    epochs = config.epochs

    in_channels = config.in_channels
    out_channels = config.out_channels
    kernel_heights = config.kernel_heights
    stride = config.stride
    padding = config.padding
    keep_probab = config.keep_probab


    if model_name == 'CNN':
        model = CNN.CNN(batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, word_embeddings)

    elif model_name == 'LSTM':
        model = LSTM_Attn.AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)



    loss_fn = F.cross_entropy
    path = "Saved Models/"
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, train_iter, epoch, loss_fn)
        val_loss, val_acc, y_test, y_pred = eval_model(model, valid_iter, loss_fn)
        _, f, o = helper.getResult(y_test, y_pred)
        current_f1 = f['f1-score']
        checkpoint_model(model, path, current_f1, epoch+1, model_name, 'max')
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

    
    load_saved_model(model, path + '{}_best.pth'.format(model_name))
    test_loss, test_acc, y_test, y_pred = eval_model(model, test_iter, loss_fn)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

    print("                                Overall               #               Fake                ")
    print("                   precision    recall      f1-score  #  precision    recall      f1-score")
    _, f, o = helper.getResult(y_test, y_pred)
    res = helper.printResult(model_name,o,f)
    print(res)
    path = model_name+"_results.txt"
    helper.saveResults(path, res)



def cnn(args):
    run_model('CNN')

def lstm(args):
    run_model('LSTM')





parser = argparse.ArgumentParser(description='Argparse!')
subparsers = parser.add_subparsers()

parser_p = subparsers.add_parser('CNN')
parser_p.set_defaults(func=cnn)

parser_q = subparsers.add_parser('LSTM')
parser_q.set_defaults(func=lstm)


parser.add_argument("-g","--gpu", action="store_true")
args = parser.parse_args()
gpu = args.gpu
if not gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

args.func(args)

