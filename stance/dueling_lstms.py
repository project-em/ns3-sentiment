from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from data_util import load_data
from article_utils import read_data_file

import numpy as np

def main():
	# Load neutral data, label as 0
	sentences = read_data_file()
	y = np.zeros(len(sentences))

	# Load liberal data, label as 1

	# Load conservative data, label as -1

	# Train liberal vs. neutral

	# Train conservative vs. neutral

	# Test


if __name__ == '__main__':
	main()