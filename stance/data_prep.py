import csv
import glob
import json
import nltk
import os
from enum import Enum
from typing import Tuple, List

# Filenames to write training data
cons_train_file = "data/training/conservative.dat"
lib_train_file = "data/training/liberal.dat"

# Filenames to write testing data
cons_test_file = "data/testing/conservative.dat"
lib_test_file = "data/testing/liberal.dat"
neutral_test_file = "data/testing/neutral.dat"

# Filenames to write validation data
cons_valid_file = "data/valid/conservative.dat"
lib_valid_file = "data/valid/liberal.dat"
neutral_valid_file = "data/valid/neutral.dat"

# Conservative or liberal model types for loading data
class DataPurpose(Enum):
    training = 1
    testing = 2
    validation = 3

def fetch_articles(data_purpose):
    # type: (DataPurpose) -> List[Tuple[str,str]]
    if (data_purpose == DataPurpose.training):
        datapattern = 'data/raw/570*/*.json'
    elif (data_purpose == DataPurpose.testing):
        datapattern = 'data/raw/test*/*.json'
    elif (data_purpose == DataPurpose.validation):
        datapattern = 'data/raw/valid*/*.json'
    data = glob.glob(datapattern)
    articles = []
    for i in range(len(data)):
        dataFile = open(data[i], 'r')
        contents = json.load(dataFile)
        site = contents['thread']['site']
        text = contents['text']
        articles.append((site, text))

    return articles


def build_neutral_data():
    # Load NLTK sentence tokenizer
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    # Read wikipedia articles to build neutral sentences
    data = glob.glob("data/articles/*")
    output_scaling_file = open("data/neutral_scaling.dat", "w+")
    output_test_file = open(neutral_test_file, "w+")
    output_valid_file = open(neutral_valid_file, "w+")
    count = 0

    for filename in data:
        if count > 15000:
            break
        dataFile = open(filename, 'r')
        for line in dataFile.readlines():
            if count > 15000:
                break
            line = line.replace("\n", "")
            line = line.replace("[REF]","")
            sentences = sent_detector.tokenize(line)
            for i, sentence in enumerate(sentences):
                if (i + count) < 5000:
                    output_test_file.write(sentence.strip() + os.linesep)
                elif (i + count) < 10000:
                    output_scaling_file.write(sentence.strip() + os.linesep)
                elif (i + count) < 15000:
                    output_valid_file.write(sentence.strip() + os.linesep)
                else:
                    break
            count += len(sentences)


# Writes the sentences to liberal and conservative data files
def write_to_files(data_purpose):
    # type: (DataPurpose) -> ()

    # Load NLTK sentence tokenizer
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    articles = fetch_articles(data_purpose)

    if data_purpose == DataPurpose.training:
        os.makedirs("training", exist_ok=True)
        conservative_file = open(cons_train_file, 'w+')
        liberal_file = open(lib_train_file, 'w+')
    elif data_purpose == DataPurpose.testing:
        os.makedirs("testing", exist_ok=True)
        conservative_file = open(cons_test_file, 'w+')
        liberal_file = open(lib_test_file, 'w+')
    elif data_purpose == DataPurpose.validation:
        os.makedirs("valid", exist_ok=True)
        conservative_file = open(cons_valid_file,'w+')
        liberal_file = open(lib_valid_file, 'w+')
    else:
        print("invalid data purpose, exiting")
        return

    for article in articles:
        text = article[1].replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('ADVERTISEMENT', '')
        text = text.replace('Advertisement', '')
        text = text.replace('RELATED: ', ' ')
        source = article[0]

        # tokenize sentences
        sentences = sent_detector.tokenize(text)

        for sentence in sentences:
            # Skip sentences that start with "HIDE CAPTION" or have an HTML div
            if (sentence.startswith("Hide Caption")):
                continue
            if "<div" in sentence:
                continue

            # Right now sourceId 5 is the only conservative source
            if source == "thehill.com" or source == "foxnews.com":
                conservative_file.write(sentence.strip() + os.linesep)
            else:
                liberal_file.write(sentence.strip() + os.linesep)

def build_scraped_data():

    # Open all files for writing
    os.makedirs("training", exist_ok=True)
    cons_train = open(cons_train_file, 'w+')
    lib_train = open(lib_train_file, 'w+')

    os.makedirs("testing", exist_ok=True)
    cons_test = open(cons_test_file, 'w+')
    lib_test = open(lib_test_file, 'w+')

    os.makedirs("valid", exist_ok=True)
    cons_valid = open(cons_valid_file, 'w+')
    lib_valid = open(lib_valid_file, 'w+')

    # Load NLTK sentence tokenizer
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

    with open('data/article.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        lib_count = 0
        cons_count = 0

        for article in reader:
            article_contents = article[2]
            source = article[9]

            text = article_contents.replace('\n', ' ')
            text = text.replace('\t', ' ')

            # tokenize sentences
            sentences = sent_detector.tokenize(text)

            for sentence in sentences:
                # Sources 1 and 2 are liberal, all others are conservative
                if source == '1' or source == '2':
                    lib_count += 1
                    if (lib_count <= 5000):
                        lib_test.write(sentence.strip() + os.linesep)
                    elif (lib_count <= 10000):
                        lib_valid.write(sentence.strip() + os.linesep)
                    else:
                        lib_train.write(sentence.strip() + os.linesep)
                else:
                    cons_count += 1
                    if (cons_count <= 5000):
                        cons_test.write(sentence.strip() + os.linesep)
                    elif (cons_count <= 10000):
                        cons_valid.write(sentence.strip() + os.linesep)
                    else:
                        cons_train.write(sentence.strip() + os.linesep)


# Given the name of an article folder (conservative, liberal, or neutral) read the
# articles from that folder and split them into sentences in a file of that name.
# Writes the sentences to liberal and conservative data files
def create_combined_data(data_purpose):
    articles = fetch_articles(data_purpose)

    # TODO: rename
    training_file = open('data/training.dat', 'w+')

    for article in articles:
        text = article[1].replace('\n', ' ')
        text = text.replace('\t', ' ')
        source = article[0]

        # Load NLTK sentence tokenizer
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text)

        for sentence in sentences:
            if source == "thehill.com" or source == "foxnews.com":
                training_file.write(sentence.encode('utf-8').strip() + "\t-1" + os.linesep)
            else:
                training_file.write(sentence.encode('utf-8').strip() + "\t1" + os.linesep)

def main():
    write_to_files(DataPurpose.training)
    write_to_files(DataPurpose.testing)
    write_to_files(DataPurpose.validation)
    build_neutral_data()

if __name__ == '__main__':
    main()
