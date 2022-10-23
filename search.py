import numpy as np
import pandas as pd
import string
import re

import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from functools import reduce
from nlp import my_lemmatizer_ru, stemming
from functools import reduce
from main import run_search_engine
from autocorrect import AutoCorrector


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
        return [self.id, self.title, self.cluster,
                chartotext(self.new_char) + '\n' + sellerstotext(self.sellers)]


index = []
title_invert_index = {}
tfidf_matrix = 0
tf = 0
corrector = 0

import ast


def chartotext(characteristics):
    characteristics = ast.literal_eval(characteristics)
    if characteristics is None or len(characteristics.keys()) == 0:
        return 'Характеристики данного товара не указаны.'
    txt = 'Характеристики товара:\n'
    for ch in characteristics.keys():
        txt += f'{ch.title()} - {characteristics[ch]}\n'
    return txt


def sellerstotext(sellers):
    sellers = ast.literal_eval(sellers)
    if sellers is None or len(sellers.keys()) == 0:
        return 'Данный товар еще не был продан.'
    txt = '                            Поставщики:\n'
    for ch in sellers.keys():
        txt += f'{ch} - {sellers[ch]} ₽\n'
    return txt


def build_index():
    global index, title_invert_index, tfidf_matrix, corrector
    # считывает сырые данные и строит индекс
    df = pd.read_csv(r'C:\Users\alina\Downloads\clearLem.csv')

    for i, row in df.iterrows():
        # print(i)
        # print(*row)
        index.append(Document(*row))

    title_invert_index = build_invert_index(df['name1'].tolist())
    tfidf_matrix = tf_idf_off(df['name1'].tolist())
    corrector = AutoCorrector("./template/file.pkl")


def tf_idf_off(ls):
    global tf
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0)
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


def retrieve(query, startPrice, endPrice):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    global index, title_invert_index
    query = corrector(query)
    print(query)
    if query is None:
        query = ''
    query = process_text(query)
    processed_query = query.split()
    words_indexes = []
    # удалить одинаковые слова в запросе
    for word in processed_query:
        if word in title_invert_index.keys():
            # был иф убрали
            words_indexes.append(title_invert_index[word])
    if len(words_indexes) == 0:
        return [], '---', 0, 0
    else:
        candidates = list(set.intersection(*map(set, words_indexes)))
        if not candidates:
            new_a = []
            for x in words_indexes: new_a += x
            candidates = new_a
        candidates = np.array(candidates)
        sc = score(query, tfidf_matrix[candidates])

        ans = np.append(sc, candidates.reshape(-1, 1), axis=1)
        b = torch.unique(torch.tensor(ans), dim=0).tolist()
        b.sort(reverse=True)
        index1 = np.array(index)

        ind, category = run_search_engine(query)

        # category_ans(b, ind)

        if startPrice is not None or endPrice is not None:
            new_data = index1[list(map(int, [x[1] for x in b]))]

            relev = del_all(new_data, startPrice, endPrice)
            finn = []
            k = 0
            if relev[0].id in ind:
                finn.append(relev[0])
                k+=1
                while len(finn) < 16 and k<len(relev):
                    if relev[k].id in ind:
                        finn.append(relev[k])
                    k += 1
            else:
                k=1
                while len(finn) < 16 and k<len(relev):
                    if len(finn) == 1:
                        finn.append(relev[0])
                        continue
                    if relev[1].id in ind:
                        finn.append(relev[k])
                    k += 1

            if not finn:
                return relev[:15], category, b, ind

            return finn, category, b, ind

        finn = []
        k = 0
        if int(b[0][1]) in ind:
            finn.append(int(b[0][1]))
            k+=1
            while len(finn) < 16 and k<len(b):
                if int(b[k][1]) in ind:
                    finn.append(int(b[k][1]))
                k += 1
        else:
            k=1
            while len(finn) < 16 and k < len(b):
                if len(finn) == 1:
                    finn.append(int(b[0][1]))
                    continue
                if int(b[k][1]) in ind:
                    finn.append(int(b[k][1]))
                k += 1

        if not finn:
            return index1[list(map(int, [x[1] for x in b[:15]]))], category, b, ind

        return index1[finn], category, b, ind


def category_ans(usdata, ind):
    tmp = [int(i[1]) for i in usdata if i[1] in ind]
    index1 = np.array(index)
    return index1[tmp]


def del_all(x, startPrice, endPrice):
    new_x = []
    if startPrice == "":
        startPrice = None
    if endPrice == "":
        endPrice = None

    for i in x:
        sellers_dict = ast.literal_eval(i.sellers)
        f = False
        for comp in sellers_dict.keys():
            if endPrice is not None and startPrice is not None:
                if float(endPrice) >= float(sellers_dict[comp]) >= float(startPrice):
                    f = True
                    break
            elif endPrice is not None:
                if float(sellers_dict[comp]) <= float(endPrice):
                    f = True
                    break
            elif startPrice is not None:
                if float(sellers_dict[comp]) >= float(startPrice):
                    f = True
                    break
        if f:
            new_x.append(i)
    return new_x
