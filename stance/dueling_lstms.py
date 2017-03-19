from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import np_utils
from article_utils import read_data_file
import glob
from collections import Counter
import numpy as np
from nltk.tokenize import wordpunct_tokenize as tokenize

import numpy as np

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

	# one hot encode the words
	# this gives the arrays an extra dimension the size of the vocabulary
	x_words = np_utils.to_categorical(x_words, vocab_size + first_index)
	y_words = np_utils.to_categorical(y_words, vocab_size + first_index)
	return (x_words, y_words)

# Load data for training dueling sentence model
# filename- full path to file to read sentences from
# there should be one sentence on each line in the file
# returns a tuple of two numpy arrays- x and y
def load_training_data(filename):
	datafile = open(filename,'r')

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
		x_train_arr.append(x_words)
		y_train_arr.append(y_words)

	x_train = np.array(x_train_arr)
	y_train = np.array(y_train_arr)
	return (x_train, y_train)

def score_sentence(model, x_words):
	#TODO: test this to see what size/format
	# get a numpy array of probability predictions
	word_probs = model.predict_proba(x_words)
	# for each word, get the probability of the next word being next
	print "probability prediction shape: ", word_probs.shape

	# the probability of a sentence is the product of probabilities
	# of each word given the words that came before it.

# predict a label for a sentence as conservative, neutral, or liberal
# -1 represents conservative, 0 is neutral, and 1 is liberal
def label_sentence(conservative_model, liberal_model, sentence):
	
	x_words, _ = sentence_to_sequences(tokenize(sentence))

	#word number 3

def train_model(X, y):
	#TODO: figure out if this should be done here or in another function
	#figure out encoding of y 
	# truncate and pad input sequences
	X = sequence.pad_sequences(X, maxlen=max_sentence_length)
	y = sequence.pad_sequences(y, maxlen=max_sentence_length)

	print "x shape after one hot encoding: ", X.shape
	print "y shape after one hot encoding: ", y.shape
	
	# create the model
	#TODO: fix this (https://keras.io/getting-started/sequential-model-guide/)
	hiddenStateSize = 128
	hiddenLayerSize = 128
	word_vec_size = vocab_size + first_index
	model = Sequential()
	in_shape = (max_sentence_length, word_vec_size)
	lstm = LSTM(hiddenStateSize, return_sequences=True, input_shape=in_shape)
	model.add(lstm)
	# Using the TimeDistributed wrapper allows us to apply a layer to every 
	# slice of the sentences (output a prediction after each word.)
	model.add(TimeDistributed(Dense(hiddenLayerSize, activation='relu')))
	#TODO: add dropout layer?
	model.add(TimeDistributed(Dense(word_vec_size, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001))
	print(model.summary())
	model.fit(X, y, nb_epoch=3, batch_size=64)
	return model

# Create conservative model and liberal model and save to file
def create_and_save_models():
	#TODO: finish
	model.save('my_model.h5')

def main():
	x_neutral, y_neutral = load_training_data("data/neutral_train.txt")
	model = train_model(x_neutral, y_neutral)

	# Train liberal vs. neutral
	#   # creates a HDF5 file 'my_model.h5'
	# del model  # deletes the existing model


if __name__ == '__main__':
	main()