from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import os
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
import run_classifier
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
sys.path.insert(1, '../Helper/')
import helper, config
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("epoch",type=int)
args = parser.parse_args()

VOCAB = '../../Data/vocab.txt'
MODEL = 'model'
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
TRAIN_CSV_PATH = 'Data/Train.csv'
TEST_CSV_PATH = 'Data/Test.csv'
train = pd.read_csv(TRAIN_CSV_PATH, index_col='articleID')
test = pd.read_csv(TEST_CSV_PATH, index_col='articleID')
cols = ['label', 'content']
train = train.loc[:, cols]
test = test.loc[:, cols]
train.fillna('UNKNOWN', inplace=True)
test.fillna('UNKNOWN', inplace=True)
test, val = train_test_split(test, test_size=0.1, random_state=109)
label_list = ["0", "1"]
train_examples = [run_classifier.InputExample(guid='train', text_a=row.content, label=str(row.label)) for row in train.itertuples()]
val_examples = [run_classifier.InputExample(guid='val', text_a=row.content, label=str(row.label)) for row in val.itertuples()]
test_examples = [run_classifier.InputExample(guid='test', text_a=row.content, label=str(row.label)) for row in test.itertuples()]

# len(test_examples)
# torch.cuda.get_device_name(0)
# print(torch.cuda.is_available())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
gradient_accumulation_steps = 1
train_batch_size = 32
eval_batch_size = 32
train_batch_size = train_batch_size // gradient_accumulation_steps
output_dir = 'best model'
bert_model = 'bert-base-multilingual-uncased'
num_train_epochs = args.epoch
num_train_optimization_steps = int(math.ceil(len(train_examples) / train_batch_size)/ gradient_accumulation_steps) * num_train_epochs
cache_dir = "model"
learning_rate = 2e-5
warmup_proportion = 0.1
max_seq_length = 128
label_list = ["0","1"]
tokenizer = BertTokenizer.from_pretrained(VOCAB)
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", cache_dir=cache_dir, num_labels=2)
model.to(device)
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = run_classifier.BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

checkpoint_history = []





def evaluation(model, evaluation_data):

    tr_loss = 0
    predicted = []
    true = []

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        predicted.extend(pred_flat)
        true.extend(labels_flat)

    eval_data, eval_dataloader = get_data(evaluation_data)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = run_classifier.accuracy(logits, label_ids)
        flat_accuracy(logits, label_ids)
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    return eval_loss, eval_accuracy, true, predicted


def get_data(raw_data):

    features = run_classifier.convert_examples_to_features(raw_data, label_list, max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=train_batch_size)
    return data, dataloader


def log(mode, data_size, batch):
    run_classifier.logger.info("***** Running " + mode + " *****")
    run_classifier.logger.info("  Num examples = %d", data_size)
    run_classifier.logger.info("  Batch size = %d", batch)


def checkpoint_model(model, path_to_save, current_score, epoch, mode='max'):

    checkpoint_history.append(current_score)
    is_best = False

    if ((np.max(checkpoint_history) == current_score and mode == 'max') or
            (np.min(checkpoint_history) == current_score and mode == 'min')):
        is_best = True
        if not os.path.exists(path_to_save):
            os.makedirs(output_dir)

        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    print('Current best', max(checkpoint_history), 'after {}th epoch'.format(epoch))

    return is_best


global_step = 0
nb_tr_steps = 0
tr_loss = 0


log("Training", len(train_examples), train_batch_size)
run_classifier.logger.info("  Num steps = %d", num_train_optimization_steps)
train_data, train_dataloader = get_data(train_examples)
model.train()
for _ in trange(int(num_train_epochs), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    total_step = len(train_data) // train_batch_size
    ten_percent_step = total_step // 10
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)
        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        if step % ten_percent_step == 0:
            print("Fininshed: {:.2f}% ({}/{})".format(step/total_step*100, step, total_step))
        val_loss, val_accuracy, true, predicted = evaluation(model, val_examples)
        t, f, o = helper.getResult(true, predicted)
        current_f1 = f['f1-score']
        checkpoint_model(model, output_dir, current_f1, step + 1, 'max')
        print(f'Step: {step + 1:02}, Val. Loss: {val_loss:3f}, Val. Acc: {val_accuracy:.2f}%')


def load_saved_model(path):

    output_dir = path
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    config = BertConfig(output_config_file)
    model = BertForSequenceClassification(config, num_labels=len(label_list))
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)  # important to specific device
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    return model


model = load_saved_model(output_dir)
loss, accuracy, true, predicted = evaluation(model, test_examples)

# print("Accuracy of BERT is:", accuracy_score(true, predicted))
# print("Precision of BERT is:", precision_score(true, predicted))
# print("Recall of BERT is:", recall_score(true, predicted))
# print("F1 Score of BERT is:", f1_score(true, predicted))
# print(classification_report(true, predicted))
# conf_mat = confusion_matrix(y_true=true, y_pred=predicted)
# print('Confusion matrix:\n', conf_mat)


print("                                Overall               #               Fake                ")
print("                   precision    recall      f1-score  #  precision    recall      f1-score")
_, f, o = helper.getResult(true, predicted)
res = helper.printResult(model_name,o,f)
print(res)
path = "bert_results.txt"
helper.saveResults(path, res)