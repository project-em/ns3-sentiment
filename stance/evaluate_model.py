import numpy as np
from sklearn.metrics import precision_score
from data_prep import DataPurpose
from dueling_lstms import compute_scale_factor
from dueling_lstms import label_sentences
from dueling_lstms import reload_model
from dueling_lstms import SourceStance

def eval_predict_provenance(data_purpose):
    # Filenames of testing data
    if (data_purpose == DataPurpose.testing):
        print("using test data")
        cons_file = "data/testing/conservative.dat"
        lib_file = "data/testing/liberal.dat"
        neutral_file = "data/testing/neutral.dat"
    elif (data_purpose == DataPurpose.validation):
        print("using validation data")
        cons_file = "data/valid/conservative.dat"
        lib_file = "data/valid/liberal.dat"
        neutral_file = "data/valid/neutral.dat"

    # Load models
    cons_model, cons_vocab = reload_model(SourceStance.conservative)
    lib_model, lib_vocab = reload_model(SourceStance.liberal)

    scaling_factor = compute_scale_factor(cons_model,lib_model,cons_vocab,lib_vocab)
    lib_thresh = 40
    cons_thresh = 30

    # Label each testing sentence
    print("labeling conservative sentences")
    cons_sentences = open(cons_file).readlines()
    Y_cons_test = np.repeat(-1, len(cons_sentences))
    cons_labels = label_sentences(cons_model,
                                  lib_model,
                                  cons_vocab,
                                  lib_vocab,
                                  cons_sentences,
                                  cons_scale_factor=scaling_factor,
                                  lib_thresh=lib_thresh,
                                  cons_thresh=cons_thresh)

    # Calculate the training and testing precision scores
    cons_testing_precision = precision_score(Y_cons_test, cons_labels, average='micro')
    unique, counts = np.unique(cons_labels, return_counts=True)
    cons_counts = dict(zip(unique, counts))

    del cons_sentences
    del Y_cons_test
    del cons_labels

    # Label each testing sentence
    print("labeling liberal sentences")
    lib_sentences = open(lib_file).readlines()
    Y_lib_test = np.repeat(1, len(lib_sentences))
    lib_labels = label_sentences(cons_model,
                                 lib_model,
                                 cons_vocab,
                                 lib_vocab,
                                 lib_sentences,
                                 cons_scale_factor=scaling_factor,
                                 lib_thresh=lib_thresh,
                                 cons_thresh=cons_thresh)

    lib_testing_precision = precision_score(Y_lib_test, lib_labels, average='micro')
    unique, counts = np.unique(lib_labels, return_counts=True)
    lib_counts = dict(zip(unique, counts))

    del lib_sentences
    del Y_lib_test
    del lib_labels

    print("labeling neutral sentences")
    neutral_sentences = open(neutral_file).readlines()
    Y_neutral_test = np.repeat(0, len(neutral_sentences))
    neutral_labels = label_sentences(cons_model,
                                     lib_model,
                                     cons_vocab,
                                     lib_vocab,
                                     neutral_sentences,
                                     cons_scale_factor=scaling_factor,
                                     lib_thresh=lib_thresh,
                                     cons_thresh=cons_thresh)

    neutral_testing_precision = precision_score(Y_neutral_test, neutral_labels, average='micro')
    unique, counts = np.unique(neutral_labels, return_counts=True)
    neutral_counts = dict(zip(unique, counts))


    print("Neutral testing precision: ", str(neutral_testing_precision))
    print("Neutral testing counts: ", str(neutral_counts))
    print("Conservative testing precision: ", str(cons_testing_precision))
    print("Conservative prediction counts: ", cons_counts)
    print("Liberal testing precision: ", str(lib_testing_precision))
    print("Liberal prediction counts: ", lib_counts)

def main():
    eval_predict_provenance(DataPurpose.testing)

if __name__ == '__main__':
    main()