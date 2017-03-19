from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from article_utils import read_data_file
import glob
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
	return (x_words, y_words)

# Load data for training dueling sentence model
# filename- full path to file to read sentences from
# there should be one sentence on each line in the file
# returns a tuple of two numpy arrays- x and y
def load_training_data(filename):
	datafile = open(filename,'r')

	print "loading sentences"
	lines = dataFile.readlines()

	all_words = dict()

	# Count the frequency of each word in the dataset
	word_counts = Counter()
	# We have to tokenize the words to do this so keep the tokenized version
	word_lines = []
	for line in lines:
		words = tokenize(line)
		word_lines.append(words)
		word_counts(word) += 1

	# free up some space
	del lines

	# Create a dictionary of the vocabulary where each word is associated
	# with an integer.
	# Using integer representations instead of full words when training the 
	# model will save space.
	word_indices = [first_index : vocab_size + first_index]
	vocab = dict(zip(word_counts.most_common(vocab_size), word_indices))

	x_train_arr = []
	y_train_arr = []
	for words in word_lines:
		(x_words, y_words) = sentence_to_sequences(words)
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
	X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
	y_train = sequence.pad_sequences(y_train, maxlen=max_sentence_length)

	# one hot encode the output word
	# this gives the y array an extra dimension the size of the vocabulary
	y_train = keras.utils.to_categorical(y_train)
	
	# create the model
	embedding_vector_length = 32
	#TODO: fix this (https://keras.io/getting-started/sequential-model-guide/)
	model = Sequential()
	model.add()
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	print(model.summary())
	model.fit(X_train, y_train, nb_epoch=3, batch_size=64)

# Create conservative model and liberal model and save to file
def create_and_save_models():


def main():
	


if __name__ == '__main__':
	main()