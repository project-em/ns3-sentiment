import numpy as np
from sklearn.metrics import precision_score
from dueling_lstms import reload_model
from dueling_lstms import label_sentences
from dueling_lstms import SourceStance
from collections import Counter

def eval_labeled_data():
    # Load training and testing splits
    sentences_training = open('data/training.dat', 'r')
    sentences_testing = open('data/testing.dat', 'r')

    # Load models
    cons_model, cons_vocab = reload_model(SourceStance.conservative)
    lib_model, lib_vocab = reload_model(SourceStance.liberal)

    Y_train = []
    train_predictions = []

    # Label each training sentence
    for line in sentences_training.readlines():
        split = line.split("\t")
        Y_train.append(int(split[1]))
        sentence = split[0]
        label = label_sentence(cons_model, lib_model, cons_vocab, lib_vocab, sentence)
        train_predictions.append(label)

    Y_test = []
    test_predictions = []

    # Label each testing sentence
    # for line in sentences_testing.readlines():
    #     split = line.split("\t")
    #     Y_test.append(int(split[1]))
    #     sentence = split[0]
    #     label = label_sentence(cons_model, lib_model, cons_model, lib_vocab, sentence)
    #     test_predictions.append(label)

    # Calculate the training and testing precision scores
    training_precision = precision_score(Y_train, train_predictions)
    testing_precision = precision_score(Y_test, test_predictions)

    print("Training precision: ", str(training_precision))
    print("Testing precision: ", str(testing_precision))

def eval_predict_provenance():
    # Filenames of testing data
    cons_test_file = "data/testing/conservative.dat"
    lib_test_file = "data/testing/liberal.dat"

    # Load models
    cons_model, cons_vocab = reload_model(SourceStance.conservative)
    lib_model, lib_vocab = reload_model(SourceStance.liberal)

    # Label each testing sentence
    print("labeling conservative sentences")
    cons_sentences = open(cons_test_file).readlines()
    Y_cons_test = np.repeat(-1, len(cons_sentences))
    cons_labels = label_sentences(cons_model, lib_model, cons_vocab, lib_vocab, cons_sentences)

    # Calculate the training and testing precision scores
    cons_testing_precision = precision_score(Y_cons_test, cons_labels, average='micro')
    unique, counts = np.unique(cons_labels, return_counts=True)
    cons_counts = dict(zip(unique, counts))

    # Label each testing sentence
    print("labeling liberal sentences")
    lib_sentences = open(lib_test_file).readlines()
    Y_lib_test = np.repeat(1, len(lib_sentences))
    lib_labels = label_sentences(cons_model, lib_model, cons_vocab, lib_vocab, lib_sentences)

    lib_testing_precision = precision_score(Y_lib_test, lib_labels, average='micro')
    unique, counts = np.unique(lib_labels, return_counts=True)
    lib_counts = dict(zip(unique, counts))

    print("Conservative testing precision: ", str(cons_testing_precision))
    print("Conservative article, predicted liberal: ", cons_counts)
    print("Liberal testing precision: ", str(lib_testing_precision))
    print("Liberal prediction counts: ", lib_counts)

def main():
    eval_predict_provenance()

if __name__ == '__main__':
    main()