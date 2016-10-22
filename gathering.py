from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json


def credentials():
    file = open("credentials.txt", "r").read()
    creds = file.split("\n")
    return creds


class StdOutListener(StreamListener):
    _isRetweet = None

    def on_data(self, data):
        try:
            tweet = json.loads(data)
        except ValueError as e:
            print("Error: " + e)

        if 'text' in tweet:
            if self._isRetweet:
                print(tweet["text"])
            else:
                if "RT @" not in tweet["text"]:
                    print(tweet["text"])

        elif 'delete' in tweet:
            print("DELETED: " + tweet["delete"])
        else:
            print(tweet)
        return True

    def on_error(self, status):
        print(status)


def streamer(subjects, isRetweet=True):
    l = StdOutListener()
    auth = OAuthHandler(credentials()[0], credentials()[1])
    auth.set_access_token(credentials()[2], credentials()[3])
    l._isRetweet = isRetweet
    stream = Stream(auth, l)
    stream.filter(track=subjects)


# streamer(["trump", "DonaldTrump", "DumpTrump"], True)
