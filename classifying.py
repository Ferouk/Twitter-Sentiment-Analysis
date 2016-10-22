from __future__ import absolute_import, print_function

import pickle
import random
from statistics import mode

from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize


def load_pickle(pickled):
    file = open("pickles/" + pickled + ".pickle", "rb")
    data = pickle.load(file)
    file.close()
    return data


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents = load_pickle("documents")
word_features = load_pickle("word_features5k")
featuresets = load_pickle("featuresets")
random.shuffle(featuresets)
print(len(featuresets))
testing_set = featuresets[10000:]
training_set = featuresets[:10000]


def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


classifier = load_pickle("originalnaivebayes5k")
MNB_classifier = load_pickle("MNB_classifier5k")
BernoulliNB_classifier = load_pickle("BernoulliNB_classifier5k")
LogisticRegression_classifier = load_pickle("LogisticRegression_classifier5k")
SGDClassifier_classifier = load_pickle("SGDClassifier_classifier5k")
SVC_classifier = load_pickle("SVC_classifier5k")
LinearSVC_classifier = load_pickle("LinearSVC_classifier5k")
NuSVC_classifier = load_pickle("NuSVC_classifier5k")

voted_classifier = VoteClassifier(
    classifier,
    LinearSVC_classifier,
    NuSVC_classifier,
    SVC_classifier,
    SGDClassifier_classifier,
    MNB_classifier,
    BernoulliNB_classifier,
    LogisticRegression_classifier)


def sentiment(text):
    feats = find_features(text, word_features)
    results = voted_classifier.classify(feats),voted_classifier.confidence(feats)
    print(results)
    return results

# sentiment("Sentence here")
