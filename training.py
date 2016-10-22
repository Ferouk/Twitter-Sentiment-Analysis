from __future__ import absolute_import, print_function

import pickle
import random

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, NuSVC, SVC


def pickle_it(dumping, file):
    save = open("pickles/" + file + ".pickle", "wb")
    pickle.dump(dumping, save)
    save.close()


def prepare_features(pos_file="positive", neg_file="positive", neut_file="neutral"):
    pos = open("output/" + pos_file, "r").read()
    neg = open("output/" + neg_file, "r").read()
    neut = open("output/" + neut_file, "r").read()

    documents = []

    for r in pos.split('\n'):
        documents.append((r, "pos"))

    for r in neg.split('\n'):
        documents.append((r, "neg"))

    for r in neut.split('\n'):
        documents.append((r, "neut"))

    pickle_it(documents, "documents")

    all_words = []

    pos_words = word_tokenize(pos)
    neg_words = word_tokenize(neg)
    neut_words = word_tokenize(neut)

    for w in pos_words:
        all_words.append(w.lower())

    for w in neg_words:
        all_words.append(w.lower())

    for w in neut_words:
        all_words.append(w.lower())

    all_words = nltk.FreqDist(all_words)

    word_features = list(all_words.keys())[:5000]
    pickle_it(word_features, "word_features5k")
    return word_features, documents


def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def prepare_datasets(documents, word_features):
    featuresets = [(find_features(tweet, word_features), sent) for (tweet, sent) in documents]
    pickle_it(featuresets, "featuresets")
    random.shuffle(featuresets)

    training_set = featuresets[:10000]
    testing_set = featuresets[10000:]

    return training_set, testing_set


def classifiers_bagging(training_set, testing_set):
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
    classifier.show_most_informative_features(15)
    pickle_it(classifier, "originalnaivebayes5k")

    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)
    pickle_it(MNB_classifier, "MNB_classifier5k")

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    print("BernoulliNB_classifier accuracy percent:",
          (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)
    pickle_it(BernoulliNB_classifier, "BernoulliNB_classifier5k")

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier accuracy percent:",
          (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)
    pickle_it(BernoulliNB_classifier, "LogisticRegression_classifier5k")

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print("SGDClassifier_classifier accuracy percent:",
          (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)
    pickle_it(SGDClassifier_classifier, "SGDClassifier_classifier5k")

    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set)) * 100)
    pickle_it(SVC_classifier, "SVC_classifier5k")

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)
    pickle_it(LinearSVC_classifier, "LinearSVC_classifier5k")

    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)
    print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)
    pickle_it(NuSVC_classifier, "NuSVC_classifier5k")


def train_it(pos_file="positive", neg_file="negative", neut_file="neutral"):
    word_features, documents = prepare_features(pos_file, neut_file, neg_file)
    training_set, testing_set = prepare_datasets(documents, word_features)
    classifiers_bagging(testing_set, testing_set)

# train_it()
