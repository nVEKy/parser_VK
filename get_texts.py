import os
import pandas as pd
from re import sub
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
import ratelim

import requests
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
import nltk

texts = []

for file_name in os.listdir('texts/psychology'):
    with open('texts/psychology/'+file_name, 'r', encoding='utf-8') as file:
        text = sub(r'\n+', ' ', file.read())
        text = sub(r' +', ' ', text)
        texts.append(text)
        
pd_texts = pd.DataFrame(columns=['text', 'theme'])

theme = []
for i in range(len(texts)):
    theme.append('psychology')

for file_name in os.listdir('texts/advert'):
    with open('texts/advert/'+file_name, 'r', encoding='utf-8') as file:
        text = sub(r'\n+', ' ', file.read())
        text = sub(r' +', ' ', text)
        texts.append(text)

pd_texts['text']=texts
for i in range(len(os.listdir('texts/advert'))):
    theme.append('advert')
pd_texts['theme']=theme
pd_texts.to_csv('class/ads_class.csv', index=False)


texts = []

for file_name in os.listdir('texts/psychology'):
    with open('texts/psychology/'+file_name, 'r', encoding='utf-8') as file:
        text = sub(r'\n+', ' ', file.read())
        text = sub(r' +', ' ', text)
        texts.append(text)
        
pd_texts = pd.DataFrame(columns=['text', 'theme'])

theme = []
for i in range(len(texts)):
    theme.append('psychology')

for file_name in os.listdir('texts/not_psychology'):
    with open('texts/not_psychology/'+file_name, 'r', encoding='utf-8') as file:
        text = sub(r'\n+', ' ', file.read())
        text = sub(r' +', ' ', text)
        texts.append(text)

pd_texts['text']=texts
for i in range(len(os.listdir('texts/not_psychology'))):
    theme.append('not_psychology')
pd_texts['theme']=theme
pd_texts.to_csv('class/psy_class.csv', index=False)


ads_texts = pd.read_csv('class/ads_class.csv')
ads_texts = ads_texts.astype({"text": str})

# Получение списка всех слов в корпусе
def get_corpus(data):
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus

# Получение текстовой строки из списка слов
def str_corpus(corpus):
    str_corpus = ''
    for i in corpus:
        str_corpus += ' ' + i
    str_corpus = str_corpus.strip()
    return str_corpus

# Получение облака слов
def get_wordCloud(corpus):
    wordCloud = WordCloud(background_color='white',
                              stopwords=STOPWORDS,
                              width=3000,
                              height=2500,
                              max_words=200,
                              random_state=42
                         ).generate(str_corpus(corpus))
    return wordCloud

corpus = get_corpus(ads_texts['text'].values)
procWordCloud = get_wordCloud(corpus)
fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.imshow(procWordCloud)
plt.axis('off')

plt.subplot(1, 2, 1)

# nltk.download("stopwords")

from nltk.corpus import stopwords
from string import punctuation

russian_stopwords = stopwords.words("russian")

# Удаление знаков пунктуации из текста
def remove_punct(text):
    #table = str.maketrans('', '', string.punctuation)
    table = {33: ' ', 34: ' ', 35: ' ', 36: ' ', 37: ' ', 38: ' ', 39: ' ', 40: ' ', 41: ' ', 42: ' ', 43: ' ', 44: ' ', 45: ' ', 46: ' ', 47: ' ', 58: ' ', 59: ' ', 60: ' ', 61: ' ', 62: ' ', 63: ' ', 64: ' ', 91: ' ', 92: ' ', 93: ' ', 94: ' ', 95: ' ', 96: ' ', 123: ' ', 124: ' ', 125: ' ', 126: ' '}
    return text.translate(table)

ads_texts['text_clean'] = ads_texts['text'].map(lambda x: x.lower())
     

ads_texts['text_clean'] = ads_texts['text_clean'].map(lambda x: remove_punct(x))
     

ads_texts['text_clean'] = ads_texts['text_clean'].map(lambda x: x.split(' '))
     

ads_texts['text_clean'] = ads_texts['text_clean'].map(lambda x: [token for token in x if token not in russian_stopwords\
                                                                  and token != " " \
                                                                  and token.strip() not in punctuation])
     

ads_texts['text_clean'] = ads_texts['text_clean'].map(lambda x: ' '.join(x))
     
corpus_clean = get_corpus(ads_texts['text_clean'].values)
procWordCloud = get_wordCloud(corpus_clean)
     
fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.imshow(procWordCloud)
plt.axis('off')

plt.subplot(1, 2, 1)

df_ads = ads_texts[['text_clean', 'theme']]

df_ads.to_csv('class/clean_text_ads.csv', index=False)



psy_texts = pd.read_csv('class/psy_class.csv')
psy_texts = psy_texts.astype({"text": str})

# Получение списка всех слов в корпусе
def get_corpus(data):
    corpus = []
    for phrase in data:
        for word in phrase.split():
            corpus.append(word)
    return corpus

# Получение текстовой строки из списка слов
def str_corpus(corpus):
    str_corpus = ''
    for i in corpus:
        str_corpus += ' ' + i
    str_corpus = str_corpus.strip()
    return str_corpus

# Получение облака слов
def get_wordCloud(corpus):
    wordCloud = WordCloud(background_color='white',
                              stopwords=STOPWORDS,
                              width=3000,
                              height=2500,
                              max_words=200,
                              random_state=42
                         ).generate(str_corpus(corpus))
    return wordCloud

corpus = get_corpus(psy_texts['text'].values)
procWordCloud = get_wordCloud(corpus)
fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.imshow(procWordCloud)
plt.axis('off')

plt.subplot(1, 2, 1)

# Удаление знаков пунктуации из текста
def remove_punct(text):
    #table = str.maketrans('', '', string.punctuation)
    table = {33: ' ', 34: ' ', 35: ' ', 36: ' ', 37: ' ', 38: ' ', 39: ' ', 40: ' ', 41: ' ', 42: ' ', 43: ' ', 44: ' ', 45: ' ', 46: ' ', 47: ' ', 58: ' ', 59: ' ', 60: ' ', 61: ' ', 62: ' ', 63: ' ', 64: ' ', 91: ' ', 92: ' ', 93: ' ', 94: ' ', 95: ' ', 96: ' ', 123: ' ', 124: ' ', 125: ' ', 126: ' '}
    return text.translate(table)

psy_texts['text_clean'] = psy_texts['text'].map(lambda x: x.lower())
     

psy_texts['text_clean'] = psy_texts['text_clean'].map(lambda x: remove_punct(x))
     

psy_texts['text_clean'] = psy_texts['text_clean'].map(lambda x: x.split(' '))
     

psy_texts['text_clean'] = psy_texts['text_clean'].map(lambda x: [token for token in x if token not in russian_stopwords\
                                                                  and token != " " \
                                                                  and token.strip() not in punctuation])
     

psy_texts['text_clean'] = psy_texts['text_clean'].map(lambda x: ' '.join(x))
     
corpus_clean = get_corpus(psy_texts['text_clean'].values)
procWordCloud = get_wordCloud(corpus_clean)
     
fig = plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.imshow(procWordCloud)
plt.axis('off')

plt.subplot(1, 2, 1)

df_psy = psy_texts[['text_clean', 'theme']]

df_psy.to_csv('class/clean_text_psy.csv', index=False)