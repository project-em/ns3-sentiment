import os
import sys
import psycopg2
import nltk.data
from keras.models import load_model

PASSWORD = os.getenv("PASSWORD")
USER = os.getenv("USER")
DATABASE = os.getenv("DATABASE")
PORT = os.getenv("PORT")
HOST = os.getenv("HOST")

# Connection to the SQL table
connection = psycopg2.connect(
    host = HOST,
    database = DATABASE,
    port = PORT,
    user = USER,
    password = PASSWORD
)


# Writes the sentences to liberal and conservative data files
def write_to_files():
	articles = fetch_articles()

	conservative_file = open('data/conservative.dat', 'w+')
	liberal_file = open('data/liberal.dat', 'w+')

	for row in articles:
		text = row[2]
		sourceId = row[8]

        # Load NLTK sentence tokenizer
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		sentences = sent_detector.tokenize(text.decode('utf-8'))


		for sentence in sentences:
			# Right now sourceId 5 is the only conservative source
			if int(sourceId) != 5:
				liberal_file.write(sentence.encode('utf-8') + os.linesep)
			else:
				conservative_file.write(sentence.encode('utf-8') + os.linesep)


# Labels sentences with bias and stores them into the SQL table
def label_sentences():
	cur = connection.cursor()

	# Load Keras model here
	# model = load_model('my_model.h5')

	articles = fetch_articles();

	for row in articles:
		text = row[2]
		articleId = row[0]

        # Load NLTK sentence tokenizer
		sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		sentences = sent_detector.tokenize(text.decode('utf-8'))

		for sentence in sentences:
			# Label sentence here using our model
			score = 0.1

            # Store the sentence in the SQL table
			cur.execute("INSERT INTO sentence (" + r'"text", "bias", "createdAt", "updatedAt", "articleId"' + ") VALUES (%s, %s, NOW(), NOW(), %s)", (sentence, str(score), str(articleId)))

	connection.commit()

# Fetches the articles from the SQL table
def fetch_articles():
	cur = connection.cursor()

	cur.execute("SELECT * FROM article;")
	return cur.fetchall()


# Run program with 'label' to label sentences and put them in the DB
# Run program with 'write' to write sentences to conservative and liberal data files
def main():
	if (len(sys.argv) > 1):
		if (sys.argv[1] == "label"):
			label_sentences()
		elif (sys.argv[1] == "write"):
			write_to_files()
		else:
			print "Please supply an argument of \'label\' or \'write\'"
	else:
		print "Please supply an argument of \'label\' or \'write\'"

if __name__ == '__main__':
	main()