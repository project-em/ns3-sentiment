import os
import psycopg2
import nltk.data
from dueling_lstms import compute_scale_factor, label_sentences, reload_model, SourceStance
from dotenv import load_dotenv, find_dotenv
from topic_scoring import compute_sim, load_word2vec

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
        user = 'smgqifpopvmfqg',
        password = PASSWORD
    )

    return connection

# Labels sentences with bias and stores them into the SQL table
def label_database_sentences():
    word2vec = load_word2vec()

    cons_model, cons_vocab = reload_model(SourceStance.conservative)
    lib_model, lib_vocab = reload_model(SourceStance.liberal)

    connection = connect()
    cur = connection.cursor()

    scaling_factor = compute_scale_factor(cons_model, lib_model, cons_vocab, lib_vocab)

    # the threshold is based on the best threshold from the evaluation script
    lib_thresh = 40
    cons_thresh = 30

    print("Fetching topics")
    topic_map = fetch_topics()
    print("topic map is:", topic_map)

    print("fetching articles")
    articles = fetch_unlabeled_articles()

    print(len(articles))
    print("labeling all sentences")
    for row in articles:
        text = row[2]
        articleId = row[0]
        topicId = row[8]
        topic = topic_map[topicId]

        print("labeling one article")
        # Load NLTK sentence tokenizer and run it on the article
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text)
        print(sentences)
        if (len(sentences) > 0):
            # Label sentence here using our dueling models
            bias_labels = label_sentences(cons_model=cons_model,
                                     lib_model=lib_model,
                                     cons_vocab=cons_vocab,
                                     lib_vocab=lib_vocab,
                                     sentences=sentences,
                                     cons_scale_factor=scaling_factor,
                                     lib_thresh=lib_thresh,
                                     cons_thresh=cons_thresh)

            topic_scores = compute_sim(sentences=sentences, topic=topic, word2vec=word2vec)

            for sentence, (bias_label, bias_score), topic_score in zip(sentences, bias_labels, topic_scores):
                # Store the sentence in the SQL table
                cur.execute("INSERT INTO sentence ("
                            + r'"text", "bias", "createdAt", "updatedAt", "articleId", "topicRelevance", "biasScore"'
                            + ") VALUES (%s, %s, NOW(), NOW(), %s, %s, %s)",
                            (sentence, str(bias_label), str(articleId), str(topic_score), str(bias_score)))

    connection.commit()

# Fetches the articles that have not yet been split into sentences
def fetch_unlabeled_articles():
    connection = connect()
    cur = connection.cursor()
    cur.execute("SELECT * FROM article WHERE id IN "
            + "(SELECT * FROM((SELECT id AS id from article)"
            + "EXCEPT(SELECT DISTINCT articleId AS id from sentence)))")
    articles = cur.fetchall()
    return articles

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

def fetch_topics():
    connection = connect()
    cur = connection.cursor()

    cur.execute("SELECT * FROM topic;")
    topics = cur.fetchall()
    topic_map = {row[0]:row[1] for row in topics}
    return topic_map

# Run program with 'label' to label sentences and put them in the DB
# Run program with 'write' to write sentences to conservative and liberal data files
def main():
    label_database_sentences()

if __name__ == '__main__':
    main()
