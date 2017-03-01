from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from data_util import load_data

import numpy as np

def main():
	# Load data
	num_words, (X_train, y_train), (X_test, y_test) = load_data()

	# Train liberal vs. neutral

	# Train conservative vs. neutral


if __name__ == '__main__':
	main()