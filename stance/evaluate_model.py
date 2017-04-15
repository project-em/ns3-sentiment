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

    #Files to write out scores to
    cons_kde_file = "data/cons_kde.txt"
    lib_kde_file = "data/lib_kde.txt"
    neutral_kde_file = "data/neutral_kde.txt"

    # Load models
    cons_model, cons_vocab = reload_model(SourceStance.conservative)
    lib_model, lib_vocab = reload_model(SourceStance.liberal)

    scaling_factor = compute_scale_factor(cons_model,lib_model,cons_vocab,lib_vocab)
    lib_thresh = 40
    cons_thresh = 30

    # Label each testing sentence
    print("labeling conservative sentences")
    cons_sentences = open(cons_file).readlines()
    cons_results = label_sentences(cons_model,
                                  lib_model,
                                  cons_vocab,
                                  lib_vocab,
                                  cons_sentences,
                                  cons_scale_factor=scaling_factor,
                                  lib_thresh=lib_thresh,
                                  cons_thresh=cons_thresh)

    # Calculate the training and testing precision scores
    cons_labels = [result[0] for result in cons_results]
    cons_scores = [result[1] for result in cons_results]
    unique, counts = np.unique(cons_labels, return_counts=True)
    cons_counts = dict(zip(unique, counts))
    with open (cons_kde_file, 'w') as f:
        for score in cons_scores:
            f.write(score + "\n")

    del cons_sentences
    del cons_results
    del cons_labels

    # Label each testing sentence
    print("labeling liberal sentences")
    lib_sentences = open(lib_file).readlines()
    lib_results = label_sentences(cons_model,
                                 lib_model,
                                 cons_vocab,
                                 lib_vocab,
                                 lib_sentences,
                                 cons_scale_factor=scaling_factor,
                                 lib_thresh=lib_thresh,
                                 cons_thresh=cons_thresh)

    lib_labels = [result[0] for result in lib_results]
    lib_scores = [result[1] for result in lib_results]
    unique, counts = np.unique(lib_labels, return_counts=True)
    lib_counts = dict(zip(unique, counts))
    with open(lib_kde_file, 'w') as f:
        for score in lib_scores:
            f.write(score + "\n")
    del lib_results
    del lib_sentences
    del lib_labels

    print("labeling neutral sentences")
    neutral_sentences = open(neutral_file).readlines()
    neutral_results = label_sentences(cons_model,
                                     lib_model,
                                     cons_vocab,
                                     lib_vocab,
                                     neutral_sentences,
                                     cons_scale_factor=scaling_factor,
                                     lib_thresh=lib_thresh,
                                     cons_thresh=cons_thresh)
    neutral_labels = [result[0] for result in neutral_results]
    neutral_scores = [result[1] for result in neutral_results]
    unique, counts = np.unique(neutral_labels, return_counts=True)
    neutral_counts = dict(zip(unique, counts))
    with open(neutral_kde_file, 'w') as f:
        for score in neutral_scores:
            f.write(score + "\n")

    print("Neutral testing counts: ", str(neutral_counts))
    print("Conservative prediction counts: ", cons_counts)
    print("Liberal prediction counts: ", lib_counts)

def main():
    eval_predict_provenance(DataPurpose.testing)

if __name__ == '__main__':
    main()