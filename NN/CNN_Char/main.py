import tensorflow as tf
import json
import sys
import numpy as np

from sklearn import metrics

from NN.CNN_Char.data_utils import Data
from NN.CNN_Char.models.char_cnn_zhang import CharCNNZhang
from NN.CNN_Char.models.char_cnn_kim import CharCNNKim
from NN.CNN_Char.models.char_tcn import CharTCN


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


tf.flags.DEFINE_string("model", "char_cnn_zhang", "Specifies which model to use: char_cnn_zhang or char_cnn_kim")
FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

FLAGS(sys.argv)

if __name__ == "__main__":
    # Load configurations
    config = json.load(open("config.json"))
    # Load training data
    data = Data(alphabet=config["data"]["alphabet"],
                input_size=config["data"]["input_size"],
                num_of_classes=config["data"]["num_of_classes"])
    data.load_data()
    training_inputs, training_labels = data.get_all_data("train")
    # # Load validation data
    # validation_data = Data(alphabet=config["data"]["alphabet"],
    #                        input_size=config["data"]["input_size"],
    #                        num_of_classes=config["data"]["num_of_classes"])
    # validation_data.load_data()
    validation_inputs, validation_labels = data.get_all_data("test")

    # Load model configurations and build model
    if FLAGS.model == "kim":
        model = CharCNNKim(input_size=config["data"]["input_size"],
                           alphabet_size=config["data"]["alphabet_size"],
                           embedding_size=config["char_cnn_kim"]["embedding_size"],
                           conv_layers=config["char_cnn_kim"]["conv_layers"],
                           fully_connected_layers=config["char_cnn_kim"]["fully_connected_layers"],
                           num_of_classes=config["data"]["num_of_classes"],
                           dropout_p=config["char_cnn_kim"]["dropout_p"],
                           optimizer=config["char_cnn_kim"]["optimizer"],
                           loss=config["char_cnn_kim"]["loss"])
    elif FLAGS.model == 'tcn':
        model = CharTCN(input_size=config["data"]["input_size"],
                        alphabet_size=config["data"]["alphabet_size"],
                        embedding_size=config["char_tcn"]["embedding_size"],
                        conv_layers=config["char_tcn"]["conv_layers"],
                        fully_connected_layers=config["char_tcn"]["fully_connected_layers"],
                        num_of_classes=config["data"]["num_of_classes"],
                        dropout_p=config["char_tcn"]["dropout_p"],
                        optimizer=config["char_tcn"]["optimizer"],
                        loss=config["char_tcn"]["loss"])
    else:
        model = CharCNNZhang(input_size=config["data"]["input_size"],
                             alphabet_size=config["data"]["alphabet_size"],
                             embedding_size=config["char_cnn_zhang"]["embedding_size"],
                             conv_layers=config["char_cnn_zhang"]["conv_layers"],
                             fully_connected_layers=config["char_cnn_zhang"]["fully_connected_layers"],
                             num_of_classes=config["data"]["num_of_classes"],
                             threshold=config["char_cnn_zhang"]["threshold"],
                             dropout_p=config["char_cnn_zhang"]["dropout_p"],
                             optimizer=config["char_cnn_zhang"]["optimizer"],
                             loss=config["char_cnn_zhang"]["loss"])
    # Train model
    model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                checkpoint_every=config["training"]["checkpoint_every"])

    # loss, accuracy, f1_score, precision, recall, = model.test(testing_inputs=validation_inputs,
    #                                                           testing_labels=validation_labels,
    #                                                           batch_size=config["training"]["batch_size"])
    # print(loss, accuracy, f1_score, precision, recall)

    # print(validation_labels)
    #
    y_predicted = np.argmax(model.test(testing_inputs=validation_inputs,
                                       testing_labels=validation_labels,
                                       batch_size=config["training"]["batch_size"]), axis=1)

    # print(y_predicted)
    # print(data.test_true)

    printScore(data.test_true, y_predicted)
