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

# Load data for dueling LSTMs
def load_dueling_data(filename):
	datafile = open(filename,'r')

	print "loading sentences"
	Xlines = dataFile.readlines()

	split = len(Xlines / 4)

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
 		Xlines += dataFile.readlines()
 
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
  
 	return word_index, (x_train, y_train), (x_test, y_test)
 	pos_indices = np.nonzero(y_test==1)
 	x_test_pos = x_test[pos_indices]
 	y_test_pos = y_test[pos_indices]
 
 	neg_indices = np.nonzero(y_test!=1)
 	x_test_neg = x_test[neg_indices]
 	y_test_neg = y_test[neg_indices]
 
 	return word_index, (x_train, y_train), (x_test_pos, y_test_pos), (x_test_neg, y_test_neg)
  
