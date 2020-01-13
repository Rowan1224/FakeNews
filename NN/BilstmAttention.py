import os, sys

from sklearn import metrics

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.client import device_lib
import numpy as np

import pandas as pd
from sklearn.utils import shuffle

# from utils.prepare_data import *
import time


# from utils.model_helper import *

def make_train_feed_dict(model, batch):
    """make train feed dict for training"""
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: .5}
    return feed_dict


def make_test_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: 1.0}
    return feed_dict


def run_train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'train_op': model.train_op,
        'loss': model.loss,
        'global_step': model.global_step,
    }
    return sess.run(to_return, feed_dict)


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

    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    true = report['1']
    fake = report['0']

    overall = {"Accuracy": metrics.accuracy_score(y_true, y_pred), "Recall": metrics.recall_score(y_true, y_pred),
               "F1-Score": metrics.f1_score(y_true, y_pred)}
    return true, fake, overall


def run_eval_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    prediction = sess.run(model.prediction, feed_dict)
    acc = np.sum(np.equal(prediction, batch[1])) / len(prediction)
    return acc, prediction, batch[1]


def get_attn_weight(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)


class ABLSTM(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]
        self.pretrained_embeddings = config["embedding"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

    def build_graph(self):
        print("building graph")
        # Word embedding
        embeddings_var = tf.Variable(self.pretrained_embeddings)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)

        rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                BasicLSTMCell(self.hidden_size),
                                inputs=batch_embedded, dtype=tf.float32)

        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len)))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, self.keep_prob)

        # Fully connected layer（dense layer)
        FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
        y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)
        class_weights = tf.constant([0.75, .25])
        weighted_logits = tf.multiply(y_hat, class_weights)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=weighted_logits, labels=self.label))

        # self.loss = tf.reduce_mean(
        #     tf.nn.weighted_cross_entropy_with_logits(logits= y_hat, targets=self.label, pos_weight=class_weights))

        # prediction
        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")


def fill_feed_dict(data_X, data_Y, batch_size):
    """Generator to yield batches"""
    # Shuffle data first.
    shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch


if __name__ == '__main__':

    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.visible_device_list = '0'
    device_name = sys.argv
    print(device_lib.list_local_devices())

    # load data
    df = pd.read_csv("../Data/Corpus/AllDataTarget.csv")
    X = df["news"].values
    Y = df["label"].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=109)

    # data preprocessing
    puncList = ["।", "”", "“", "’"]
    x = "".join(puncList)
    filterString = x + '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n০১২৩৪৫৬৭৮৯'
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000, filters=filterString)
    tokenizer.fit_on_texts(x_train)
    train_idx = tokenizer.texts_to_sequences(x_train)
    test_idx = tokenizer.texts_to_sequences(x_test)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(train_idx, maxlen=32, padding='post', truncating='post')
    x_test = tf.keras.preprocessing.sequence.pad_sequences(test_idx, maxlen=32, padding='post', truncating='post')
    vocab_size = 50000
    print("train size: ", len(x_train))
    print("vocab size: ", vocab_size)

    # split dataset to test and dev
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * 0.1)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]

    print("Validation Size: ", dev_size)

    word_index = tokenizer.word_index
    embeddings_index = {}
    for i, line in enumerate(open('/media/MyDrive/Project Fake news/Models/cc.bn.300.vec')):
        val = line.split()
        embeddings_index[val[0]] = np.asarray(val[1:], dtype='float32')

    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_matrix = np.asarray(embedding_matrix, dtype='float32')

    config = {
        "max_len": 32,
        "hidden_size": 256,
        "vocab_size": vocab_size,
        "embedding_size": 300,
        "n_class": 2,
        "learning_rate": 2e-3,
        "batch_size": 32,
        "train_epoch": 25,
        "embedding": embedding_matrix
    }

    classifier = ABLSTM(config)
    classifier.build_graph()

    sess = tf.Session(config=config_gpu)
    sess.run(tf.global_variables_initializer())
    dev_batch = (x_dev, y_dev)
    start = time.time()
    true_df = []
    fake_df = []
    overall_df = []
    for e in range(config["train_epoch"]):

        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, config["batch_size"]):
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
            attn = get_attn_weight(classifier, sess, (x_batch, y_batch))
            # plot the attention weight
            # print(np.reshape(attn, (config["batch_size"], config["max_len"])))
        t1 = time.time()

        print("Train Epoch time:  %.3f s" % (t1 - t0))
        dev_acc, pred, true = run_eval_step(classifier, sess, dev_batch)
        print("validation accuracy: %.3f " % dev_acc)

        print("Training finished, time consumed : ", time.time() - start, " s")
        print("Start evaluating:  \n")
        cnt = 0
        test_acc = 0
        y_pred = []
        y_true = []
        for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
            acc, pred, true = run_eval_step(classifier, sess, (x_batch, y_batch))
            y_pred.extend(pred)
            y_true.extend(true)
            test_acc += acc
            cnt += 1

        t, f, o = printScore(y_true, y_pred)
        t["epoch"] = e
        f["epoch"] = e
        o["epoch"] = e
        true_df.append(t)
        fake_df.append(f)
        overall_df.append(o)
        print("Test accuracy : %f %%" % (test_acc / cnt * 100))

    df = pd.DataFrame(true_df)
    df.to_csv("../Data/results/true.csv", index=None, header=True)
    df = pd.DataFrame(fake_df)
    df.to_csv("../Data/results/fake.csv", index=None, header=True)
    df = pd.DataFrame(overall_df)
    df.to_csv("../Data/results/overall.csv", index=None, header=True)

