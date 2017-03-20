import json
import numpy as np
from collections import Counter
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import np_utils
from nltk.tokenize import wordpunct_tokenize as tokenize

# Variables used for both training and scoring sentences
max_sentence_length = 50
vocab_size = 8000
sentence_end = 0
sentence_start = 1
unknown_token = 2
first_index = 3


# Use the vocabulary to turn a sentence into two integer vectors, the
# X vector which is the sentence beginning with the sentence_start token, 
# and the y vector which is the sentence shifted left by one
# vocab- a mapping of words to integers
# words- an array of words representing a sentence
# note: this means tokenization should be done before calling this method
# returns a tuple of the x sentence and the y sentence
def sentence_to_sequences(vocab, words):
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
    datafile = open(filename, 'r')

    print "loading sentences"
    lines = datafile.readlines()

    all_words = dict()

    # Count the frequency of each word in the dataset
    word_counts = Counter()
    # We have to tokenize the words to do this so keep the tokenized version
    word_lines = []
    for line in lines:
        words = tokenize(line)
        word_lines.append(words)
        for word in words:
            word_counts[word] += 1

    # free up some space
    del lines

    # Create a dictionary of the vocabulary where each word is associated
    # with an integer.
    # Using integer representations instead of full words when training the
    # model will save space.
    word_indices = range(first_index, vocab_size + first_index)
    vocab = dict(zip(word_counts.most_common(vocab_size), word_indices))

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


def train_model(X, y):
    # truncate and pad input sequences
    X = sequence.pad_sequences(X, maxlen=max_sentence_length)
    y = sequence.pad_sequences(y, maxlen=max_sentence_length)

    print "x shape after one hot encoding: ", X.shape
    print "y shape after one hot encoding: ", y.shape

    # model parameters
    hiddenStateSize = 128
    hiddenLayerSize = 128
    batch_size = 64
    epochs = 1
    word_vec_size = vocab_size + first_index

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
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))
    print(model.summary())

    # Train the model
    model.fit(X, y, nb_epoch=epochs, batch_size=batch_size)
    return model


# Create conservative model and liberal model and save to file
def create_and_save_models():
    # Get conservative word encoding
    cons_train_file = "data/conservative_train.txt"
    x_cons, y_cons, vocab_cons = load_training_data(cons_train_file)
    # save vocab for encoding test sentences later
    with open("conservative_vocab.json") as f:
        json.dump(vocab_cons, f)
    del vocab_cons

    # train conservative model on conservative article data
    conservative_model = train_model(x_cons, y_cons)
    conservative_model.save('conservative_model.h5')
    del conservative_model

    # Get liberal word encoding
    lib_train_file = "data/liberal_train.txt"
    x_lib, y_lib, vocab_lib = load_training_data(lib_train_file)
    # save vocab for encoding test sentences later
    with open("liberal_vocab.json") as f:
        json.dump(vocab_lib, f)
    del vocab_lib

    # train liberal model on liberal article data
    liberal_model = train_model(x_lib, y_lib)
    liberal_model.save('liberal_model.h5')
    del liberal_model


# SCORING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Score the likelihood of a sentence under a given model
# Computes the likelihood of each word in the sentence given the previous
# words and multiplies these probabilities to get an overall probability
def score_sentence(model, x_words_encoded, y_words):
    # TODO: test this to see what size/format
    # get a numpy array of probability predictions
    word_probs = model.predict_proba(x_words_encoded)
    # for each word, get the probability of the next word being next
    print "probability prediction shape: ", word_probs.shape
    print "prob prediction: ", word_probs

    word_predict = model.predict(x_words_encoded)
    print "word prediction shape: ", word_predict.shape
    print "word predictions: ", word_predict

    # the probability of a sentence is the product of probabilities
    # of each word given the words that came before it.

    # TODO: we use log probabilities and sum them so we don't get underflow



# predict a label for a sentence as conservative, neutral, or liberal
# -1 represents conservative, 0 is neutral, and 1 is liberal
def label_sentence(cons_model, lib_model, cons_vocab, lib_vocab, sentence):
    x_cons, y_cons = sentence_to_sequences(cons_model, tokenize(sentence))

    # one hot encode the input sentence
    x_cons = np_utils.to_categorical(x_cons, vocab_size + first_index)

    cons_score = score_sentence(cons_model, x_cons, y_cons)



    # TODO: write intermediate output to file or DB?

    # word number 3

# TODO: remove this testing code
def main():
    x_neutral, y_neutral, vocab = load_training_data("data/neutral_train.txt")
    model = train_model(x_neutral, y_neutral)
    model.save('conservative_model.h5')

    datafile = open("data/neutral_test.txt", 'r')

    print "loading sentences"
    lines = datafile.readlines()
    label_sentence(model, None, vocab, None, lines[0])

if __name__ == '__main__':
    main()
