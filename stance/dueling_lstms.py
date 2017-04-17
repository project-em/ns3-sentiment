import json
import numpy as np
import os
from collections import Counter
from enum import Enum
from keras.layers import Dense, Dropout, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import np_utils
from nltk.tokenize import wordpunct_tokenize as word_tokenize
from typing import Dict, Tuple, List, Generator

# Conservative or liberal model types for loading data
class SourceStance(Enum):
    conservative = 1
    liberal = 2

# Filenames of training data
cons_train_file = "data/training/conservative.dat"
lib_train_file = "data/training/liberal.dat"

# Filenames in which to save/load models
model_dir = "models/"
cons_model_file = model_dir + "conservative_model.h5"
cons_vocab_file = model_dir + "conservative_vocab.json"
lib_model_file = model_dir + "liberal_model.h5"
lib_vocab_file = model_dir + "liberal_vocab.json"

# Variables used for both training and scoring sentences
max_sentence_length = 35
min_sent_chars = 20
vocab_size = 8000
sentence_end = 0
sentence_start = 1
unknown_token = 2
first_index = 3
word_vec_size = vocab_size + first_index

# Use the vocabulary to turn a sentence into two integer vectors, the
# X vector which is the sentence beginning with the sentence_start token, 
# and the y vector which is the sentence shifted left by one
# vocab- a mapping of words to integers
# words- an array of words representing a sentence
# note: this means tokenization should be done before calling this method
# returns a tuple of the x sentence and the y sentence
def sentence_to_sequences(vocab, words):
    # type: (Dict[str, int], List[str]) -> Tuple[List[int], List[int]]
    x_words = [sentence_start]
    y_words = []
    for word in words:
        if word in vocab:
            x_words.append(vocab[word])
            y_words.append(vocab[word])
        else:
            x_words.append(unknown_token)
            y_words.append(unknown_token)

    x_words.append(sentence_end)
    y_words.append(sentence_end)

    return (x_words, y_words)


#TODO: document
def arrayize_sentences(vocab, sentences, hot_encode_y = True):
    # type: (Dict[str, int], List[str], bool) -> Tuple[np.array, np.array]
    x_batch_arr = []
    y_batch_arr = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        (x_words, y_words) = sentence_to_sequences(vocab, words)

        # one hot encode the words
        # this gives the arrays an extra dimension the size of the vocabulary
        x_words = np_utils.to_categorical(x_words, vocab_size + first_index)
        if hot_encode_y:
            y_words = np_utils.to_categorical(y_words, vocab_size + first_index)

        x_batch_arr.append(x_words)
        y_batch_arr.append(y_words)

    x_batch = np.array(x_batch_arr)
    y_batch = np.array(y_batch_arr)
    return x_batch, y_batch

# TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create dictionary of words for training sentence model
# filename- full path to file to read sentences from
# there should be one sentence on each line in the file
# returns a tuple containing the sentences in the file
# and the dictionary mapping words to int encodings
def create_vocab(filename):
    # type: (str) -> Tuple[int, Dict[str, int]]

    print("creating vocabulary")
    datafile = open(filename, 'r')
    all_sentences = [line.lower().strip() for line in datafile.readlines()]

    sentences = []
    for sentence in all_sentences:
        sentence = sentence.lower().strip()
        # only train on lines longer than a min number of chars
        if len(sentence) > min_sent_chars:
            sentences.append(sentence)

    # Count the frequency of each word in the dataset
    word_counts = Counter()
    # We have to tokenize the words to do this so keep the tokenized version
    word_lines = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        word_lines.append(words)
        for word in words:
            word_counts[word] += 1

    # Create a dictionary of the vocabulary where each word is associated
    # with an integer.
    # Using integer representations instead of full words when training the
    # model will save space.
    common_words = [comm[0] for comm in word_counts.most_common(vocab_size)]
    word_indices = range(first_index, vocab_size + first_index)
    return len(sentences), dict(zip(common_words, word_indices))

# Generator for batches of data for training
# Necessary because training on all the data at once leads to memory issues
def gen_training_data(filename, vocab, batch_size):
    # type: (str, Dict[str, int], int) -> Generator[Tuple[np.array,np.array], None, None]

    datafile = open(filename, 'r')
    while(True):

        # get a batch of sentences
        batch_sentences = []
        this_batch_size = 0
        while (this_batch_size < batch_size):
            sentence = datafile.readline().lower().strip()
            # only train on lines longer than a min number of chars
            # but need to keep sentences that are empty for the batch
            # to finish at the end of the file
            if (len(sentence) > min_sent_chars) or sentence == "":
                batch_sentences.append(sentence)
                this_batch_size += 1

        x_batch, y_batch = arrayize_sentences(vocab, batch_sentences, hot_encode_y=True)

        x_batch = sequence.pad_sequences(x_batch,
                               maxlen=max_sentence_length,
                               padding='post',
                               truncating='post')
        y_batch = sequence.pad_sequences(y_batch,
                               maxlen=max_sentence_length,
                               padding='post',
                               truncating='post')
        yield x_batch, y_batch

# Define the LSTM layers and train the model
def train_model(filename, num_lines, vocab):
    # type: (str, int, Dict[str, int]) -> Sequential

    print("training model")

    # model parameters
    hiddenStateSize = 256
    hiddenLayerSize = 256
    batch_size = 64
    epochs = 3
    learning_rate = 0.01

    # create the model
    model = Sequential()
    in_shape = (max_sentence_length, word_vec_size)
    model.add(Dropout(0.4, input_shape=in_shape))
    lstm = LSTM(hiddenStateSize, return_sequences=True)
    model.add(lstm)
    # Using the TimeDistributed wrapper allows us to apply a layer to every
    # slice of the sentences (output a prediction after each word.)
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(hiddenLayerSize, activation='relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(word_vec_size, activation='softmax')))
    # loss function categorical because each possible next word is a category
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate))
    print(model.summary())

    # Train the model
    num_samples = int(num_lines / batch_size) * batch_size
    generator = gen_training_data(filename=filename, vocab=vocab, batch_size=batch_size)
    model.fit_generator(generator, nb_epoch=epochs, samples_per_epoch=num_samples)
    return model


# Create conservative model and liberal model and save to file
def create_and_save_models():
    os.makedirs(model_dir, exist_ok=True)
    # Get conservative word encoding
    n_cons, vocab_cons = create_vocab(cons_train_file)

    # save vocab for encoding test sentences later
    with open(cons_vocab_file, 'w') as f:
        json.dump(vocab_cons, f)

    # train conservative model on conservative article data
    conservative_model = train_model(cons_train_file, n_cons, vocab_cons)
    del vocab_cons
    conservative_model.save(cons_model_file)
    del conservative_model

    # Get liberal word encoding
    n_lib, vocab_lib = create_vocab(lib_train_file)
    # save vocab for encoding test sentences later
    with open(lib_vocab_file, 'w') as f:
        json.dump(vocab_lib, f)

    # train liberal model on liberal article data
    liberal_model = train_model(lib_train_file, n_lib, vocab_lib)
    del vocab_lib
    liberal_model.save(lib_model_file)
    del liberal_model


# SCORING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Score the likelihood of each sentence in a batch under a given model
# Computes the likelihood of each word in the sentence given the previous
# words and multiplies these probabilities to get an overall probability
# assumes X sentences are already one hot encoded but y sentences are not
# this function does all computations in batch in order to run faster
def score_sentences(model, X_hot, y):
    # type: (Sequential, np.array, np.array) -> float

    # assume X sentences are already one hot encoded
    # truncate and pad sequences
    # num sentences * max sentence length * vocabulary size sized array
    X_hot = sequence.pad_sequences(X_hot,
                                   maxlen=max_sentence_length,
                                   padding='post',
                                   truncating='post')

    # num sentences * max sentence length sized array
    y = sequence.pad_sequences(y,
                               maxlen=max_sentence_length,
                               padding='post',
                               truncating='post')

    # get a numpy array of probability predictions
    # num sentences * max sentence length * vocabulary size
    print("getting predictions from model")
    word_probs = model.predict_proba(X_hot)
    # use log probabilities so we don't get underflow
    word_probs = np.log(word_probs)
    # the probability of a sentence is the product of probabilities
    # of each word given the words that came before it.
    print("computing overall probabilities for each sentence")
    sentence_probs = np.zeros(y.shape[0])
    for sentence_pos in range(max_sentence_length):
        # get the probabilities of each possible word being the next word
        # results in a num_sentences * vocab size array
        word_pos_probs = word_probs[:, sentence_pos]
        # get the index within the vocab of the next word in each sentence
        # results in a num sentences sized array
        vocab_indices = y[:, sentence_pos]
        # get the probabilities of only the actual next word
        next_word_probs = np.array([word_pos_probs[i,w] for i,w in enumerate(vocab_indices)])
        # the product of probabilities is the sum of log probabilities
        sentence_probs = np.add(sentence_probs, next_word_probs)

    print(sentence_probs)
    return sentence_probs

# takes a group of sentences, formats them one by one into the correct data format
# and then scores them in a batch in order to run faster
# then labels them one by one as liberal or conservative based on the score
def label_sentences(cons_model, lib_model, cons_vocab, lib_vocab, sentences,
                    lib_thresh = 30, cons_thresh = 30, cons_scale_factor = 1):
    # type: (Sequential, Sequential, Dict[str, int], Dict[str, int], List[str]) -> List[(int,float)]

    sentences = [sentence.lower().strip() for sentence in sentences]
    X_cons, y_cons = arrayize_sentences(cons_vocab, sentences, hot_encode_y=False)
    cons_scores = score_sentences(cons_model, X_cons, y_cons)
    cons_scores = cons_scores * cons_scale_factor

    X_lib, y_lib = arrayize_sentences(lib_vocab, sentences, hot_encode_y=False)
    lib_scores = score_sentences(lib_model, X_lib, y_lib)

    sentence_labels = []
    for cons_score, lib_score in zip(cons_scores, lib_scores):
        label = 0
        diff = cons_score - lib_score
        if diff > cons_thresh:
            label = 1 # label conservative
        elif diff < -lib_thresh:
            label = -1 # label liberal
        sentence_labels.append((label,diff))

    print("number of sentence labels is: ", len(sentence_labels))
    return sentence_labels

# Computes the average ratio of liberal to conservative scores on neutral data.
# Multiply the conservative scores times this factor to even them out
def compute_scale_factor(cons_model, lib_model, cons_vocab, lib_vocab):
    neutral_file = "data/neutral_scaling.dat"

    sentences = [sentence.lower().strip() for sentence in open(neutral_file).readlines()]

    X_cons, y_cons = arrayize_sentences(cons_vocab, sentences, hot_encode_y=False)
    cons_scores = score_sentences(cons_model, X_cons, y_cons)

    X_lib, y_lib = arrayize_sentences(lib_vocab, sentences, hot_encode_y=False)
    lib_scores = score_sentences(lib_model, X_lib, y_lib)

    lib_to_cons_ratios = np.divide(lib_scores, cons_scores)
    lib_to_cons_ratios = lib_to_cons_ratios[np.nonzero(np.isfinite(lib_to_cons_ratios))]
    print("lib to cons ratios are: ", lib_to_cons_ratios)
    scaling_factor = np.mean(lib_to_cons_ratios)
    print ("computed scaling factor is: ", scaling_factor)
    return scaling_factor

# Load a model and the vocab encoding needed to test the probability
# of a sentence within that model.
def reload_model(stance):
    # type: (SourceStance) -> Tuple[Sequential, Dict[str, int]]
    if stance == SourceStance.conservative:
        model = load_model(cons_model_file)
        with open(cons_vocab_file, 'r') as f:
            vocab = json.load(f)
        return (model, vocab)
    else:
        model = load_model(lib_model_file)
        with open(lib_vocab_file, 'r') as f:
            vocab = json.load(f)
        return (model, vocab)

def main():
    create_and_save_models()

if __name__ == '__main__':
    main()
