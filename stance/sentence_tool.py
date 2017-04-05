import os
import sys
import psycopg2
import nltk.data
from dueling_lstms import compute_scale_factor, label_sentences, reload_model, SourceStance
from dotenv import load_dotenv, find_dotenv

def connect():
    load_dotenv(find_dotenv())
    PASSWORD = os.getenv("PASSWORD")
    DATABASE = os.getenv("DATABASE")
    PORT = os.getenv("PORT")
    HOST = os.getenv("HOST")

    print(PASSWORD)

    # Connection to the SQL table
    connection = psycopg2.connect(
        host = HOST,
        database = DATABASE,
        port = PORT,
        user = 'pxwgadeeffddil',
        password = PASSWORD
    )

    return connection

# Labels sentences with bias and stores them into the SQL table
def label_database_sentences():
    cons_model, cons_vocab = reload_model(SourceStance.conservative)
    lib_model, lib_vocab = reload_model(SourceStance.liberal)

    connection = connect()
    cur = connection.cursor()

    scaling_factor = compute_scale_factor(cons_model, lib_model, cons_vocab, lib_vocab)
    # TODO: set threshold based on best threshold from evaluation script
    lib_thresh = 30
    cons_thresh = 30

    print("fetching articles")
    articles = fetch_articles()

    print(len(articles))
    print("labeling all sentences")
    for row in articles:
        text = row[2]
        articleId = row[0]

        print("labeling one article")
        # Load NLTK sentence tokenizer and run it on the article
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text)

        # Label sentence here using our dueling models
        labels = label_sentences(cons_model=cons_model,
                                 lib_model=lib_model,
                                 cons_vocab=cons_vocab,
                                 lib_vocab=lib_vocab,
                                 sentences=sentences,
                                 cons_scale_factor=scaling_factor,
                                 lib_thresh=lib_thresh,
                                 cons_thresh=cons_thresh)

        for sentence, label in zip(sentences, labels):
            # Store the sentence in the SQL table
            cur.execute("INSERT INTO sentence ("
                        + r'"text", "bias", "createdAt", "updatedAt", "articleId"'
                        + ") VALUES (%s, %s, NOW(), NOW(), %s)",
                        (sentence, str(label), str(articleId)))

    connection.commit()

# Fetches the articles from the SQL table
def fetch_articles():
    connection = connect()
    cur = connection.cursor()

    cur.execute("SELECT * FROM article WHERE " + r'"archivalDataFlag"' + "=0;")
    articles = cur.fetchall()
    return articles

# Fetches the articles from the SQL table
def fetch_sentences():
    connection = connect()
    cur = connection.cursor()

    cur.execute("SELECT * FROM sentence;")
    return cur.fetchall()


# Run program with 'label' to label sentences and put them in the DB
# Run program with 'write' to write sentences to conservative and liberal data files
def main():
    label_database_sentences()

if __name__ == '__main__':
    main()
