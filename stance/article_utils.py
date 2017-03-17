import re

# Clean up an article and then split it into a list of sentences
def split_sentences(article):
	# remove Wikipedia "[REF]" and replace unreadable characters with readable versions
	article = article.replace(" [REF]", "")
	article = article.replace("[REF]", "")
	article = article.replace("\xe2\x80\x9c", "\"")
	article = article.replace("\xe2\x80\x9d", "\"")
	article = article.replace("\xe2\x80\x98", "'")
	article = article.replace("\xe2\x80\x99", "'")
	article = article.replace("\xe2\x80\x93", "-")
	article = article.replace("\s.\s", ".\s")

	# replace newlines within sentence because newline will be the delimeter in the file
	sentences = [sentence.strip().replace("\n", " ")
			for sentence in re.split("(?<=[a-z0-9)\"'])\.\s", article) 
			if not sentence.isspace()]
	return sentences

def read_data_file(datafile):
	splitlines = datafile.read().splitlines()
	sentences = [line.decode('utf-8', 'ignore') for line in splitlines]
	return sentences