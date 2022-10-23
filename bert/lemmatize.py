import pymorphy2
import string
from tqdm import tqdm
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from joblib import Parallel, delayed

NUM_WORKERS = 16


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def generate(stop_words, samples, i):
    samples = [*split(samples, NUM_WORKERS)][i]
    lemmatizer = pymorphy2.MorphAnalyzer()
    processed = []
    
    for i in tqdm(samples):
        i = i.lower()
        i = [i.strip(string.punctuation) for i in i.split(' ')]
        i = [i for i in i if i not in stop_words]
        i = " ".join(i)

        i = i.split()
        i = ' '.join([lemmatizer.parse(word)[0].normal_form
                        for word in i])

        processed.append(i)

    return processed


df = pd.read_csv("/home/vadim/K/hack/dataset/train_bert.csv")
samples = df["input"].values
print('Dataset loaded')

stop_words = set(stopwords.words('russian'))
stop_words.remove('без')
stop_words.remove('с')
cleaned_parts = Parallel(n_jobs=NUM_WORKERS, backend='threading')(delayed(generate)(stop_words,samples, i) for i in range(NUM_WORKERS))

cleaned = []
for i in cleaned_parts: cleaned += i
df["input"] = cleaned
df.to_csv("train_cleaned.csv")