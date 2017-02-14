from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import glob
import numpy as np
import enchant

engl_dict = enchant.Dict("en_US")
# TODO: deal with punctuation
def correct_words(words):
	corrected_words = []
	for word in words:
		if engl_dict.check(word):
			corrected_words.append(word)
		else: 
			corrections = engl_dict.suggest(word)
			one_word_corrs = filter(lambda w: (" " not in w) and ("-" not in w), corrections)
			if len(one_word_corrs) > 0:
				corrected_words.append(one_word_corrs[0])
	return corrected_words

def load_data():
	datapattern = 'data/abortion/*.data'
	data = glob.glob(datapattern)

	metapattern = 'data/abortion/*.meta'
	meta = glob.glob(metapattern)

	#load labels
	print "loading labels"
	Y = []
	for i in range(len(meta)):
		metaFile = open(meta[i], 'r')
		line = metaFile.readline()
		line = metaFile.readline()
		line = metaFile.readline()
		line = line.split("Stance=")[1]
		y = int(line.split("\n")[0])
		if y == 1:
			Y.append(y)
		else:
			Y.append(0)

	split = len(Y) / 4

	y_train = np.array(Y[:split])
	y_test = np.array(Y[split:])

	print "y_train is: ", y_train
	print "y_test is: ", y_test

	print "total number of sentences is: ", len(Y)

	#load sentence data
	print "loading sentences"
	Xlines = []
	for i in range(len(data)):
		dataFile = open(data[i], 'r')
		Xlines.append(dataFile.readline())

	Xlines_train = Xlines[:split]
	Xlines_test = Xlines[split:]
	print "test lines are: ", Xlines_test

	all_words = dict()
	word_index = 1

	x_train_arr = []
	for line in Xlines_train:
		words = filter(lambda w: len(w) > 0, line.split(" "))
		corrected_words = correct_words(words)
		int_words = []
		for word in corrected_words:
			if word not in all_words:
				all_words[word] = word_index
				word_index += 1
			int_words.append(all_words[word])
		x_train_arr.append(int_words)

	x_test_arr = []
	for line in Xlines_test:
		words = filter(lambda w: len(w) > 0, line.split(" "))
		corrected_words = correct_words(words)
		int_words = []
		# only add words that were seen in the training data
		# TODO: is this necessary?
		for word in corrected_words:
			if word in all_words:
				int_words.append(all_words[word])
		x_test_arr.append(int_words)

	x_train = np.array(x_train_arr)
	x_test = np.array(x_test_arr)

	pos_indices = np.nonzero(y_test==1)
	x_test_pos = x_test[pos_indices]
	y_test_pos = y_test[pos_indices]

	neg_indices = np.nonzero(y_test!=1)
	x_test_neg = x_test[neg_indices]
	y_test_neg = y_test[neg_indices]

	return word_index, (x_train, y_train), (x_test_pos, y_test_pos), (x_test_neg, y_test_neg)

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