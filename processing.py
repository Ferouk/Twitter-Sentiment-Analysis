from __future__ import absolute_import, print_function

import json
import re
import os
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from os import listdir


def stop_words(string):
    sw = [line.rstrip('\n') for line in open('assets/stop_words')]
    words = word_tokenize(string)
    words = [w for w in words if not w.lower() in sw]
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]
    words = [w for w in words if len(w) > 1]
    string = " ".join(words)
    return string


def clean_url(string):
    string = re.sub(r'(https|http)?://(\w|\.|/|\?|=|&|%)*\b', '', string, flags=re.MULTILINE)
    string = string.replace("http", "")
    string = string.replace("htt", "")
    return string


def clean_twitter(string):
    string = re.sub('@(\w{1,15})\b', '', string)
    string = string.replace("via ", "")
    string = string.replace("RT ", "")
    string = string.lower()
    return string


def clean_sc(string):
    string = re.sub('[^a-zA-Z \n]', '', string)
    string = re.sub(' +', ' ', string)
    return string


def clean_duplicates(lines):
    clean = []
    for line in lines:
        if line not in clean:
            clean.append(line)
    return clean


def parse_tweet(string):
    try:
        data = json.loads(string)
    except ValueError as e:
        print("Error: " + e)
    return data["text"]


def files_in_folder(directory):
    files = listdir(directory)
    return files


def read_file(file):
    f = open(file)
    lines = []
    for line in iter(f):
        lines.append(line)
    f.close()
    return lines


def process(input, output, twitter=True, url=True, sc=True, sw=True):
    path = "output"
    if not os.path.exists(path):
        os.makedirs(path)

    files = files_in_folder(input)
    print("Reading " + str(len(files)) + " file(s) from " + input + " directory.")
    total_line = 0
    for file in files:
        lines = read_file(input + "/" + file)
        total_line += len(lines)
        print("Parsing : " + file)
        for line in lines:
            tweet = parse_tweet(line)
            if twitter:
                tweet = clean_twitter(tweet)
            if url:
                tweet = clean_url(tweet)
            if sc:
                tweet = clean_sc(tweet)
            if sw:
                tweet = stop_words(tweet)
            if len(tweet.split(" ")) > 1:
                with open(os.path.join(path, output), 'a+') as output_file:
                    output_file.write(tweet + "\n")
                output_file.close()
    print("Total tweets " + str(total_line))

# process("data", "test")
