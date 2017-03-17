from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from data_util import load_data
from article_utils import read_data_file

import numpy as np

def train(X, Y):
	# load the dataset
	num_words, (X_train, y_train), (X_test, y_test) = load_data()
	
	# truncate and pad input sequences
	max_opinion_length = 200
	X_train = sequence.pad_sequences(X_train, maxlen=max_opinion_length)
	X_test = sequence.pad_sequences(X_test, maxlen=max_opinion_length)
	
	# create the model
	embedding_vector_length = 32
	model = Sequential()
	model.add(Embedding(num_words, embedding_vector_length, input_length=max_opinion_length))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train, nb_epoch=3, batch_size=64)

	# Final evaluation of the model
	print "testing model"
	scores = model.evaluate(X_test, y_test, verbose=0)
	predictions = model.predict_classes(X_test)
	print "predictions are: ", predictions
	print("Accuracy: %.2f%%" % (scores[1]*100))

def main():
	# Load neutral data, label as 0
	neutral_data_file = open("neutral.data", "r")
	neutral_sentences = read_data_file(neutral_data_file)
	y = np.zeros(len(neutral_sentences))

	# Load liberal data, label as 1
	liberal_data_file = open("liberal.data", "r")
	liberal_sentences = read_data_file(liberal_data_file)
	y = np.zeros(len(liberal_sentences))

	# Load conservative data, label as -1
	conservative_data_file = open("liberal.data", "r")
	conservative_sentences = read_data_file(conservative_data_file)
	y = np.zeros(len(conservative_sentences))

	# Train liberal vs. neutral
	# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
	# del model  # deletes the existing model

	# Train conservative vs. neutral

	# Test


if __name__ == '__main__':
	main()