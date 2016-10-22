from __future__ import absolute_import, print_function

import lda
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import logging


logging.getLogger("lda").setLevel(logging.WARNING)


def read(file):
    token_dict = {}
    path = 'output/' + file
    lines = [line.rstrip('\n') for line in open(path, 'r')]
    for count in range(len(lines)):
        token_dict[count] = lines[count]
    print(str(len(token_dict)) + " tweets to label.")
    return lines, token_dict


def dtm_maker(token_dict):
    print("Building DTM ...")
    tf = CountVectorizer()
    print("Fitting DTM ...")
    tfd = tf.fit_transform(token_dict.values())
    print("Obtaining the feature names")
    vocab = tf.get_feature_names()
    print(vocab)

    return tf, tfd


def topic_modeling(topic_num, tfd):
    print("Building LDA ...")
    model = lda.LDA(n_topics=topic_num, n_iter=500, random_state=1)
    print("Fitting LDA to data set ...")
    model.fit_transform(tfd)
    return model


def plotting(model):
    try:
        plt.style.use('ggplot')
    except:
        # version of matplotlib might not be recent
        pass

    doc_topic = model.doc_topic_
    f, ax = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
    for i, k in enumerate([1, 23, 44, 86, 98]):
        ax[i].stem(doc_topic[k, :], linefmt='r-',
                   markerfmt='ro', basefmt='w-')
        ax[i].set_xlim(-1, 3)
        ax[i].set_ylim(0, 1)
        ax[i].set_ylabel("Prob")
        ax[i].set_title("Tweet {}".format(k))

    ax[4].set_xlabel("Topic")

    plt.tight_layout()
    plt.show()


def export_labeled(topic1, topic2, topic3):
    print("Exporting topics to files ...")

    file_topic1 = open("output/neutral", 'a+')
    for item in topic1:
        file_topic1.write("%s\n" % item)
    file_topic1.close()

    file_topic2 = open("output/positive", 'a+')
    for item in topic2:
        file_topic2.write("%s\n" % item)
    file_topic2.close()

    file_topic3 = open("output/negative", 'a+')
    for item in topic3:
        file_topic3.write("%s\n" % item)
    file_topic3.close()


def labeling(file, topics):
    lines, token_dict = read(file)
    tf, tfd = dtm_maker(token_dict)
    model = topic_modeling(topics, tfd)

    i = 0
    topic1, topic2, topic3 = [], [], []
    for tweet in lines:
        # print("Topic " + str(model.doc_topic_[i].argmax() + 1) + " , " + tweet)
        if model.doc_topic_[i].argmax() == 0:
            topic1.append(tweet)
        elif model.doc_topic_[i].argmax() == 1:
            topic2.append(tweet)
        else:
            topic3.append(tweet)
        i += 1

    print("Total Topic 1 = " + str(len(topic1)))
    print("Total Topic 2 = " + str(len(topic2)))
    print("Total Topic 3 = " + str(len(topic3)))

    export_labeled(topic1, topic2, topic3)
    # plotting(model)

    return topic1, topic2, topic3


# topic1, topic2, topic3 = labeling("test", 3)
