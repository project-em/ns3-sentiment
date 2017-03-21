import os
import sys
import psycopg2
import nltk.data
from dueling_lstms import label_sentence, reload_model, SourceStance
from dotenv import load_dotenv, find_dotenv

def connect():
    load_dotenv(find_dotenv())
    PASSWORD = os.getenv("PASSWORD")
    DATABASE = os.getenv("DATABASE")
    PORT = os.getenv("PORT")
    HOST = os.getenv("HOST")

    print PASSWORD

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
def label_sentences():
    cons_model, cons_vocab = reload_model(SourceStance.conservative)
    lib_model, lib_vocab = reload_model(SourceStance.liberal)

    connection = connect()
    cur = connection.cursor()

    articles = fetch_articles()

    for row in articles:
        text = row[2]
        articleId = row[0]

        # Load NLTK sentence tokenizer and run it on the article
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text.decode('utf-8', 'ignore'))

        for sentence in sentences:
            # Label sentence here using our dueling models
            score = label_sentence(cons_model=cons_model,
                                   lib_model=lib_model,
                                   cons_vocab=cons_vocab,
                                   lib_vocab=lib_vocab,
                                   sentence=sentence)

            # Store the sentence in the SQL table
            cur.execute("INSERT INTO sentence ("
                        + r'"text", "bias", "createdAt", "updatedAt", "articleId"'
                        + ") VALUES (%s, %s, NOW(), NOW(), %s)",
                        (sentence, str(score), str(articleId)))

    connection.commit()

# Fetches the articles from the SQL table
def fetch_articles():
    connection = connect()
    cur = connection.cursor()

    cur.execute("SELECT * FROM article;")
    return cur.fetchall()

# Fetches the articles from the SQL table
def fetch_sentences():
    connection = connect()
    cur = connection.cursor()

    cur.execute("SELECT * FROM sentence;")
    return cur.fetchall()


# Run program with 'label' to label sentences and put them in the DB
# Run program with 'write' to write sentences to conservative and liberal data files
def main():
    label_sentences()

if __name__ == '__main__':
    main()