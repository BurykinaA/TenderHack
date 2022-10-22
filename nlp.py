from nltk import wordnet, pos_tag
from nltk import WordNetLemmatizer
import pymorphy2
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
import re

s = '!"#$%&\'()*+,./:;<=>?@[\]^_`{|}~'


def get_wordnet_pos(treebank_tag):
    my_switch = {
        'J': wordnet.wordnet.ADJ,
        'V': wordnet.wordnet.VERB,
        'N': wordnet.wordnet.NOUN,
        'R': wordnet.wordnet.ADV,
    }
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.wordnet.NOUN


def cleaning(t):
    t = t.lower()
    t = [i.strip(s) for i in t.split(' ')]
    t = [i for i in t if i not in stopwords.words('russian')]
    return " ".join(t)


def stemming(t):
    t = cleaning(t)
    stemmer = SnowballStemmer(language='russian')
    t = ' '.join([stemmer.stem(word) for word in t.split()])
    return t


def lemming(t):
    t = cleaning(t)
    lemmatizer = WordNetLemmatizer()
    tokenized_sent = t.split()
    pos_tagged = [(word, get_wordnet_pos(tag)) for word, tag in pos_tag(tokenized_sent)]
    t = ' '.join([lemmatizer.lemmatize(word, tag) for word, tag in pos_tagged])
    return t


def my_lemmatizer_ru(sent):
    sent = cleaning(sent)
    lemmatizer = pymorphy2.MorphAnalyzer()
    tokenized_sent = sent.split()
    return ' '.join([lemmatizer.parse(word)[0].normal_form
                     for word in tokenized_sent])
