import json
import numpy as np
import os
from collections import Counter
from enum import Enum
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import np_utils
from nltk.tokenize import wordpunct_tokenize as word_tokenize
from typing import Dict, Tuple, List

# Conservative or liberal model types for loading data
class SourceStance(Enum):
    conservative = 1
    liberal = 2

# Filenames of training data
cons_train_file = "data/conservative.dat"
lib_train_file = "data/liberal.dat"

# Filenames in which to save/load models
model_dir = "models/"
cons_model_file = model_dir + "conservative_model.h5"
cons_vocab_file = model_dir + "conservative_vocab.json"
lib_model_file = model_dir + "liberal_model.h5"
lib_vocab_file = model_dir + "liberal_vocab.json"

# Variables used for both training and scoring sentences
max_sentence_length = 35
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

# TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load data for training dueling sentence model
# filename- full path to file to read sentences from
# there should be one sentence on each line in the file
# returns a 3-tuple containing two numpy arrays- x and y
# and a map of vocab to the integers used to encode it
def load_training_data(filename):
    # type: (str) -> Tuple[np.array, np.array, Dict[str, int]]

    print("loading sentences")
    datafile = open(filename, 'r')
    sentences = [line.lower() for line in datafile.readlines()]

    # Count the frequency of each word in the dataset
    word_counts = Counter()
    # We have to tokenize the words to do this so keep the tokenized version
    word_lines = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        word_lines.append(words)
        for word in words:
            word_counts[word] += 1

    # free up some space
    del sentences

    # Create a dictionary of the vocabulary where each word is associated
    # with an integer.
    # Using integer representations instead of full words when training the
    # model will save space.
    common_words = [comm[0] for comm in word_counts.most_common(vocab_size)]
    word_indices = range(first_index, vocab_size + first_index)
    vocab = dict(zip(common_words, word_indices))

    x_train_arr = []
    y_train_arr = []
    for words in word_lines:
        (x_words, y_words) = sentence_to_sequences(vocab, words)

        # one hot encode the words
        # this gives the arrays an extra dimension the size of the vocabulary
        x_words = np_utils.to_categorical(x_words, vocab_size + first_index)
        y_words = np_utils.to_categorical(y_words, vocab_size + first_index)

        x_train_arr.append(x_words)
        y_train_arr.append(y_words)

    x_train = np.array(x_train_arr)
    y_train = np.array(y_train_arr)
    return (x_train, y_train, vocab)

# Define the LSTM layers and train the model
def train_model(X, y):
    # type: (np.array, np.array) -> Sequential

    print("training model")
    # truncate and pad input sequences
    X = sequence.pad_sequences(X, maxlen=max_sentence_length)
    y = sequence.pad_sequences(y, maxlen=max_sentence_length)

    print("Shape of X training array: ", X.shape)
    print("Shape of Y training array: ", y.shape)

    # model parameters
    hiddenStateSize = 128
    hiddenLayerSize = 128
    batch_size = 64
    epochs = 1

    # create the model
    model = Sequential()
    in_shape = (max_sentence_length, word_vec_size)
    lstm = LSTM(hiddenStateSize, return_sequences=True, input_shape=in_shape)
    model.add(lstm)
    # Using the TimeDistributed wrapper allows us to apply a layer to every
    # slice of the sentences (output a prediction after each word.)
    model.add(TimeDistributed(Dense(hiddenLayerSize, activation='relu')))
    # TODO: add dropout layer?
    model.add(TimeDistributed(Dense(word_vec_size, activation='softmax')))
    # loss function categorical because each possible next word is a category
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))
    print(model.summary())

    # Train the model
    model.fit(X, y, nb_epoch=epochs, batch_size=batch_size)
    return model


# Create conservative model and liberal model and save to file
def create_and_save_models():
    os.makedirs(model_dir, exist_ok=True)
    # Get conservative word encoding
    x_cons, y_cons, vocab_cons = load_training_data(cons_train_file)

    # save vocab for encoding test sentences later
    with open(cons_vocab_file, 'w') as f:
        json.dump(vocab_cons, f)
    del vocab_cons

    # train conservative model on conservative article data
    conservative_model = train_model(x_cons, y_cons)
    del x_cons
    del y_cons
    conservative_model.save(cons_model_file)
    del conservative_model

    # Get liberal word encoding
    x_lib, y_lib, vocab_lib = load_training_data(lib_train_file)
    # save vocab for encoding test sentences later
    with open(lib_vocab_file, 'w') as f:
        json.dump(vocab_lib, f)
    del vocab_lib

    # train liberal model on liberal article data
    liberal_model = train_model(x_lib, y_lib)
    del x_lib
    del y_lib
    liberal_model.save(lib_model_file)
    del liberal_model


# SCORING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Score the likelihood of a sentence under a given model
# Computes the likelihood of each word in the sentence given the previous
# words and multiplies these probabilities to get an overall probability
def score_sentence(model, x_words_encoded, y_words):
    # type: (Sequential, np.array, np.array) -> float
    # TODO: test this to see what size/format
    X = np.expand_dims(x_words_encoded, axis=0)
    X = sequence.pad_sequences(X, maxlen=max_sentence_length)
    # get a numpy array of probability predictions
    word_probs = model.predict_proba(X)
    word_probs = np.squeeze(word_probs, axis=(0,))
    # the probability of a sentence is the product of probabilities
    # of each word given the words that came before it.
    sentence_prob = 0
    for (sentence_pos, word_index) in enumerate(y_words):
        # use log probabilities and sum them so we don't get underflow
        word_pos_prob = np.log(word_probs[sentence_pos, word_index])
        sentence_prob += word_pos_prob
        #TODO: remove debug statement
        print("sentence pos is : ", sentence_pos,
              " and word index is: ", word_index,
              " and prob is: ", word_pos_prob)

    # TODO: do in batch?
    return sentence_prob

# predict a label for a sentence as conservative, neutral, or liberal
# -1 represents conservative, 0 is neutral, and 1 is liberal
def label_sentence(cons_model, lib_model, cons_vocab, lib_vocab, sentence):
    # type: (Sequential, Sequential, Dict[str, int], Dict[str, int]) -> int
    # put sentence in lowercase because training data/vocab is read in lowercase
    sentence = sentence.lower()

    # Get a prediction from the conservative model
    x_cons, y_cons = sentence_to_sequences(cons_vocab, word_tokenize(sentence))
    # one hot encode the sentence to get a score for it
    x_cons_hot = np_utils.to_categorical(x_cons, word_vec_size)
    cons_score = score_sentence(cons_model, x_cons_hot, y_cons)

    # Get a prediction from the liberal model
    x_lib, y_lib = sentence_to_sequences(lib_vocab, word_tokenize(sentence))
    # one hot encode the sentence to get a score for it
    x_lib_hot = np_utils.to_categorical(x_lib, word_vec_size)
    lib_score = score_sentence(lib_model, x_lib_hot, y_lib)

    print("conservative score is: ", cons_score,
          " and liberal score is: ", lib_score)

    # TODO: determine threshold of difference between scores and return int label
    # right now returning dummy number
    return 0

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
