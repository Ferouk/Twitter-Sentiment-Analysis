from __future__ import absolute_import, print_function

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from flask import Flask

app = Flask(__name__)


@app.route('/sentiment/<text>')
def sentiment(text):
    """ text should be processed before predicting sentiment """
    print(text)
    # sent = classify.sentiment(text)
    """ Once sentiment is predicted it should be appended to the tweet body """
    """ After preparing the tweet it should be stored into ElasticSearch """
    return text


if __name__ == '__main__':
    app.run()
