#!/usr/bin/env python

import re
import pickle
import spacy
import gensim
import os
import csv
import random
import numpy
import scipy
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sourcedir = 'russian-troll-tweets-master'
model_file = 'Russians_model'
pkl_dict = 'pos_dict.pkl'
number_of_options = 25
positive = [u'great']
negative = [u'sad']

model = gensim.models.Word2Vec.load(model_file)
pickleFile = open(pkl_dict, 'rb')
posd = pickle.load(pickleFile)

### BEGIN functions and classes
def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i+1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i+1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i+1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i+1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0]-1, mat._shape[1])

def fixSub(word, sub):
    if word.tag_[0] == 'V':
        if word.text[-1].lower() == 's':
            sub = sub + 's'
        elif word.text[-3:].lower() == 'ing':
            sub = sub + 'ing'
        elif word.text[-2:].lower() == 'ed':
            sub = sub + 'ed'
            sub = re.sub('eed$', 'ed', sub)
        elif word.text == 'is':
            sub = word.text

    if word.text[0].isupper():
        sub = sub[0].capitalize() + sub[1:]
    if word.text[1].isupper():
        nw = []
        for l in sub:
            nw.append(l.capitalize())
        sub = ''.join(nw)
    return sub

def transform_tweet(assertion):
    nlp = spacy.load('en')
    parsed = nlp(assertion)
    new_words = []
    for word in parsed:
        try:
            hits = []
            psw = word.tag_.split('__')[0]
            for item in model.wv.most_similar(positive=positive + [word.lemma_.lower()], negative=negative, topn=number_of_options):
                if posd[item[0]]:
                    psd = next(iter(posd[item[0]])).split('__')[0]
                    if (psw not in ('DT', 'PUNCT', 'IN', 'WDT')) and (psw == psd):
                        hits.append(fixSub(word, item[0]))

            if word.text in { u"is", u"was", u"be", u"\'s", u"has", u"have" }:
                new_words.append(word.text)
            elif len(hits) > 0:
                new_words.append(hits[0])
            else:
                new_words.append(word.text)
        except:
            new_words.append(word.text)
    response = ' '.join(new_words)
    response = re.sub(r'\s+([:;,\.]+)', r'\1', response)
    response = re.sub('\s+\)', '\)', response)
    response = re.sub('\(\s+', '\(', response)
    response = re.sub(r'\s([?.!,](?:\s|$))', r'\1', response)
    response = re.sub('\s+', ' ', response)
    response = re.sub(" \'s", "\'s", response)
    response = re.sub(r'# (\w)', r'#\1', response)
#    response = re.sub(r'(\s[\'\"]) (\w)', r'\1\2', response)
    response = re.sub(r'^([\'\"\xe2])\s+', r'\1', response)
    response = re.sub(r'\s+([\'\"\xe2])$', r'\1', response)
    response = re.sub(r'([\!\?\.])\s+([\!\?\.])', r'\1\2', response)
    response = re.sub(r'([Ii]) ([\"\',\xe2])m ', r'\1\2m ', response)
    response = re.sub(r'n ([\"\',\xe2])t ', r'n\1t ', response)
    response = re.sub(r'd ([\"\',\xe2])ve ', r'd\1ve ', response)
    response = re.sub(r'(\w) ([\"\',\xe2])re ', r'\1\2re ', response)
    response = re.sub(r'(\w) ([\"\',\xe2])d ', r'\1\2d ', response)
#    response = re.sub(r'(\W+)\s+(\W+)', r'\1\2', response)
    response = re.sub(r'(\')\s+([^\']+)\s+(\')', r'\1\2\3', response)
    response = re.sub(r'(\")\s+([^\"]+)\s+(\")', r'\1\2\3', response)
    response = re.sub(r'(\xe2)\s+([^\xe2]+)\s+(\xe2)', r'\1\2\3', response)
    return response

#def by_stripped_tweet_text(t):
#    return t[0]
### END functions and classes
corpus = list()

for fname in os.listdir(sourcedir):
    if fname.endswith('csv'):
        tweets = list()
        with open(os.path.join(sourcedir, fname), 'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader: tweets.append(row)

        for tweet in tweets:
            if tweet['language'] == 'English':
                t = tweet['content']
                t = re.sub('http\S*', '', t)
                t = re.sub('pic.twitter\S*', '', t)
                t = re.sub('\s+$', '', t)
                t = re.sub('^\s+', '', t)
#                corpus.append([t, tweet['content']])
                corpus.append(t)

#corpus.sort(key=by_stripped_tweet_text)
corpus.sort()
item = 1
while item < len(corpus):
#    if corpus[item][0] == corpus[item-1][0]:
    if corpus[item] == corpus[item-1]:
        corpus.pop(item)
    else:
        item = item + 1

print('Positive: ' + positive[0].encode('utf-8'))
print('Negative: ' + negative[0].encode('utf-8'))
print('Number of tweets: ' + str(len(corpus)))
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    token_pattern=r'\b\w+\b',
    min_df=1)

# vectorized_corpus = vectorizer.fit_transform([item[0] for item in corpus])
vectorized_corpus = vectorizer.fit_transform(corpus)

index = random.randint(0,len(corpus))
print('Initial tweet index: ' + str(index) + '\n')

total_words = 0
while total_words < 50000:
    selected_tweet = corpus.pop(index)
    delete_row_csr(vectorized_corpus, index)
    # response = transform_tweet(unicode(selected_tweet[0], 'utf-8'))
    response = transform_tweet(unicode(selected_tweet, 'utf-8'))
    # print(selected_tweet[1])
    print(selected_tweet)
    print(response.encode('utf-8') + '\n')
    # total_words = total_words + len(selected_tweet[1].split()) + len(response.split())
    total_words = total_words + len(selected_tweet.split()) + len(response.split())
    vectorized_response = vectorizer.transform([response])
    vector_similarity = cosine_similarity(vectorized_response, vectorized_corpus)
    index = numpy.argmax(vector_similarity)
