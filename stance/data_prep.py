import glob
import json
import nltk
import os

def fetch_articles():
    datapattern = 'data/training/570*/*.json'
    data = glob.glob(datapattern)
    print len(data)
    articles = []
    for i in range(len(data)):
        dataFile = open(data[i], 'r')
        contents = json.load(dataFile)
        site = contents['thread']['site']
        text = contents['text']
        articles.append((site, text))

    return articles

# Given the name of an article folder (conservative, liberal, or neutral) read the 
# articles from that folder and split them into sentences in a file of that name.
# Writes the sentences to liberal and conservative data files
def write_to_files():
    articles = fetch_articles()

    conservative_file = open('data/conservative.dat', 'w+')
    liberal_file = open('data/liberal.dat', 'w+')

    print articles[0][1]
    for article in articles:
        text = article[1].replace('\n', ' ')
        text = text.replace('\t', ' ')
        source = article[0]

        # Load NLTK sentence tokenizer
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text)

        for sentence in sentences:
            # Right now sourceId 5 is the only conservative source
            if source == "thehill.com" or source == "foxnews.com":
                conservative_file.write(sentence.encode('utf-8').strip() + os.linesep)
            else:
                liberal_file.write(sentence.encode('utf-8').strip() + os.linesep)


# Given the name of an article folder (conservative, liberal, or neutral) read the
# articles from that folder and split them into sentences in a file of that name.
# Writes the sentences to liberal and conservative data files
def create_combined_data():
    articles = fetch_articles()

    training_file = open('data/training.dat', 'w+')

    print articles[0][1]
    for article in articles:
        text = article[1].replace('\n', ' ')
        text = text.replace('\t', ' ')
        source = article[0]

        # Load NLTK sentence tokenizer
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text)

        for sentence in sentences:
            # Right now sourceId 5 is the only conservative source
            if source == "thehill.com" or source == "foxnews.com":
                training_file.write(sentence.encode('utf-8').strip() + "\t-1" + os.linesep)
            else:
                training_file.write(sentence.encode('utf-8').strip() + "\t1" + os.linesep)


if __name__ == '__main__':
    create_combined_data()
    #TODO: add "conservative" and "liberal" articles
