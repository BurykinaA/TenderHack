import numpy as np
import pandas as pd
import string
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from functools import reduce
from nlp import my_lemmatizer_ru, stemming
from functools import reduce


# фильтры

class Document:
    def __init__(self, id, title, cluster, new_char, sellers, title_proc):
        # что убрать что добавть
        # можете здесь какие-нибудь свои поля подобавлять
        self.id = id
        self.title = title
        self.cluster = cluster
        self.new_char = new_char
        self.sellers = sellers
        self.title_proc = title_proc

    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        # что добавлять лизе
        return [self.id, self.title, self.cluster, self.new_char]


index = []
title_invert_index = {}
tfidf_matrix = 0
tf = 0


def build_index():
    global index, title_invert_index, tfidf_matrix
    # считывает сырые данные и строит индекс
    df = pd.read_csv(r'C:\Users\alina\Downloads\clearLem.csv')

    for i, row in df.iterrows():
        # print(i)
        # print(*row)
        index.append(Document(*row))

    title_invert_index = build_invert_index(df['name1'].tolist())
    tfidf_matrix = tf_idf_off(df['name1'].tolist())


def tf_idf_off(ls):
    global tf
    tf = TfidfVectorizer()
    vectorizer = tf.fit_transform(ls)  # TfidfVectorizer(analyzer='word')
    return vectorizer


def build_invert_index(lst):
    invert_index = {}
    for i in range(len(lst)):
        for word in str(lst[i]).split():
            if word not in invert_index.keys():
                invert_index[word] = [i]
            else:
                invert_index[word].append(i)
    return invert_index


def process_text(text):
    return my_lemmatizer_ru(text)


def score(query, doc):
    query_vec = tf.transform([query])
    return cosine_similarity(doc, query_vec)


def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    global index, title_invert_index
    processed_query = process_text(query).split()
    words_indexes = []
    for word in processed_query:
        if word in title_invert_index.keys():
            if len(words_indexes) == 0:
                words_indexes.append(title_invert_index[word])
    if len(words_indexes) == 0:
        # перепроверка на транслит
        return []
    else:
        candidates = list(set.intersection(*map(set,words_indexes)))
        candidates = np.array(candidates)
        sc = score(query, tfidf_matrix[candidates])
        ans = np.append(sc, candidates.reshape(-1, 1), axis=1)
        ans.sort()

        index1 = np.array(index)
        finl_doc = index1[list(map(int, ans[:, 1][:15]))]

        return finl_doc

# import time
#
# start_time = time.time()
# build_index()
# print("Построение индекса заняло --- %s seconds ---" % (time.time() - start_time))
