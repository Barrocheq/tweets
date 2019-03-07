import re
import tensorflow as tf
import numpy as np

#Contains all line of my file without any preprocess
lines = []

#Contains words
words = []
word2int = {}
int2word = {}
vocab_size = 0

with open('/Users/Quentin/PycharmProjects/tweetClass/outputtest.txt', 'r') as file:
    lines = file.readlines()


# Get Sentiments and text
# TODO : split around ), bug if text contain )
def clean_header(line):
    headers = line.split(')')[0]
    text = line.split(')')[1].strip() # strip is use to trim the text
    text = re.sub(r'[^\w\s]', '', text)
    sentiments = headers.split(',')[1]
    serialize_sentiment = serialize_class(sentiments)
    if serialize_sentiment != -1000:
        return (serialize_sentiment, text)
    else:
        return (None, None)

def replace_text(line):
    clean_line = re.sub(r'http\S+', '', line) # remove link
    clean_line = clean_line.replace('#', '').replace('@', '') # remove hashtags and @
    clean_line = clean_line.lower()
    return clean_line

def serialize_class(sentiment):
    if sentiment == 'neg':
        return -1
    elif sentiment == 'pos':
        return 1
    elif sentiment == 'neu':
        return 0
    else:
        return -1000

def create_tab_word(line):
    if line != None:
        for word in line.split():
            if word not in words:
                words.append(word)

def word2Vec():
    for i, word in enumerate(words):
        word2int[word] = i
        int2word[i] = word
emb = []

def preprocess():
    global emb
    for line in lines:
        replace = replace_text(line)
        (sentiment, text) = clean_header(replace)
        create_tab_word(text)
    word2Vec()
    idx = sorted(word2int.values())
    eye = np.eye(max(idx) + 1)
    emb = eye[idx]


preprocess()



###################### MODEL


from tensorflow import keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])