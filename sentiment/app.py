from flask import Flask, request, abort
from stance.sentence_tool import label_article
app = Flask(__name__)

@app.route('/live', methods=["POST"])
def live_post_sentences():
    ''' article payload contains one key 'id' that points to the article ID '''
    if request.method == 'POST':
        payload = request.json
        if not payload or not payload['id']:
            abort(401)
        else:
            article_id = payload['id']
            label_article(article_id)
            return "OK"
    else:
        abort(401)
