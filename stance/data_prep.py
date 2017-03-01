from article_utils import split_sentences
import glob

# Given the name of an article folder (conservative, liberal, or neutral) read the 
# articles from that folder and split them into sentences in a file of that name.
def prep_articles(foldername):

	datapattern = 'data/' + foldername + '/*'
	article_files = glob.glob(datapattern)

	# Output all the split sentences into the same file.
	sentence_file = open('data/' + foldername + ".txt", 'w')

	for article_name in article_files:
		article_file = open(article_name, "r")
		article = article_file.read()
		
		article_file.close()
		sentences = split_sentences(article)
		for sentence in sentences:
			sentence_file.write(str(sentence) + "\n")

	sentence_file.close()

if __name__ == '__main__':
	prep_articles("neutral")
	#TODO: add "conservative" and "liberal" articles
	