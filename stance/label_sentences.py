import os
import psycopg2
import urlparse # import urllib.parse for python 3+

from keras.models import load_model

DATABASE = os.getenv("DATABASE_URL")
# result = urlparse.urlparse(DATABASE)
# username = result.username
# password = result.password
# database = result.path[1:]
# hostname = result.hostname
connection = psycopg2.connect(
    host = "ec2-54-221-217-158.compute-1.amazonaws.com",
    database = "de5h8anqo0daif",
    port = "5432",
    user = "pxwgadeeffddil",
    password = "d3cb30d58957015ab38423da03ebfa67496a385f621e27b29d8f9cd4863a0e22"
)

cur = connection.cursor()

cur.execute("SELECT * FROM sentence;")
for row in cur:
	# train row[1]
	print row
	score = 0.1
	#cur.execute("UPDATE sentence SET bias=" + score + "WHERE id="+)