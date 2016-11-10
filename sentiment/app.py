import os, logging
from flask import Flask, request, render_template, jsonify
from flask_restplus import Api, Resource

app = Flask(__name__)
logging.getLogger('flask_ask').setLevel(logging.DEBUG)

ROOT_URL = os.getenv('ROOT_URL', 'localhost')
VERSION_NO = os.getenv('VERSION_NO', '1.0')
APP_NAME = os.getenv('APP_NAME', "Devil's Advocate Sentiment")
DEBUG = os.getenv('DEBUG', False)

api = Api(app, version=VERSION_NO, title=APP_NAME)
public_ns = api.namespace('api/v1', description='Public methods')

@public_ns.route('/parse')
@public_ns.param('text', 'the text to parse')
class Parse(Resource):

    def post(text):
        # do things and stuff
        return 'temp'

if __name__ == '__main__':
    app.run()
    