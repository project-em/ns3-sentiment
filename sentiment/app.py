import os, logging, json
from flask import Flask, request, render_template, jsonify
from flask_restplus import Api, Resource, fields

app = Flask(__name__)
logging.getLogger('flask_ask').setLevel(logging.DEBUG)

ROOT_URL = os.getenv('ROOT_URL', 'localhost')
VERSION_NO = os.getenv('VERSION_NO', '1.0')
APP_NAME = os.getenv('APP_NAME', "Devil's Advocate Sentiment")
DEBUG = os.getenv('DEBUG', False)

api = Api(app, version=VERSION_NO, title=APP_NAME)
public_ns = api.namespace('api/v1', description='Public methods')

article = api.model('Article', {
    'article': fields.String(description='Article body', required=True)
})

@public_ns.route('/parse')
class Parse(Resource):

    @public_ns.expect(article)
    def post(self):
        data = json.loads(request.data)
        # do things and stuff
        return data['article'].split('.')[0]

if __name__ == '__main__':
    app.run()
    