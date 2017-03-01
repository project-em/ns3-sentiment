from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from data_util import load_data

import numpy as np

def main():
	# load the dataset
	num_words, (X_train, y_train), (x_test_pos, y_test_pos), (x_test_neg, y_test_neg) = load_data()
	# truncate and pad input sequences
	max_opinion_length = 200
	X_train = sequence.pad_sequences(X_train, maxlen=max_opinion_length)
	x_test_pos = sequence.pad_sequences(x_test_pos, maxlen=max_opinion_length)
	x_test_neg = sequence.pad_sequences(x_test_neg, maxlen=max_opinion_length)
	
	# create the model
	embedding_vector_length = 32
	model = Sequential()
	model.add(Embedding(num_words, embedding_vector_length, input_length=max_opinion_length))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train, nb_epoch=2, batch_size=64)

	# Final evaluation of the model
	print "testing model"
	pos_scores = model.evaluate(x_test_pos, y_test_pos, verbose=0)
	print("Positive Accuracy: %.2f%%" % (pos_scores[1]*100))

	neg_scores = model.evaluate(x_test_neg, y_test_neg, verbose=0)
	print("Negative Accuracy: %.2f%%" % (neg_scores[1]*100))

if __name__ == '__main__':
	main()