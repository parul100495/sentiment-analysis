"""
@pargupta
02/16/2020

"""
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import os
import re

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

from nltk.corpus import wordnet

import nltk
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')


from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import punkt
stop_words = stopwords.words('english')

from nltk.tokenize import sent_tokenize, word_tokenize

dictionary = set(nltk.corpus.words.words())

readMe = open('Code/DataSubmission/annotation2.csv','r', encoding = "cp1252").readlines() #read = r
x = []
y = []

for i in range(1,251):
    r = readMe[i].split(',')
    x.append(r[0])
    y.append(r[1])


lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def maxmatch(word, dictionary):
    if not word:
        return []
    for i in range(len(word),1,-1):
        first = word[0:i]
        rem = word[i:]
        if lemmatize(first).lower() in dictionary: #Important to lowercase lemmatized words before comparing in dictionary. 
            return [first] + maxmatch(rem, dictionary)
    first = word[0:1]
    rem = word[1:]
    return [first] + maxmatch(rem,dictionary)

x_refined = []

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
dictionary = set(nltk.corpus.words.words()) 


def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word,'n')
    return lemma


def maxmatch(word,dictionary):
    if not word:
        return []
    for i in range(len(word),1,-1):
        first = word[0:i]
        rem = word[i:]
        if lemmatize(first).lower() in dictionary: 
            return [first] + maxmatch(rem,dictionary)
    first = word[0:1]
    rem = word[1:]
    return [first] + maxmatch(rem,dictionary)
import string

def preprocess(tweet):
    
    tweet = re.sub("@\w+","",tweet).strip()
    tweet = re.sub("http\S+","",tweet).strip()
    hashtags = re.findall("#\w+",tweet)
    
    tweet = tweet.lower()
    tweet = re.sub("#\w+","",tweet).strip() 
    
    hashtag_tokens = [] 
    
    for hashtag in hashtags:
        hashtag_tokens.append(maxmatch(hashtag[1:],dictionary))        
    
    segmenter = nltk.data.load('tokenizers/punkt/english.pickle')
    segmented_sentences = segmenter.tokenize(tweet)
    
    
    processed_tweet = []
    
    word_tokenizer = nltk.tokenize.regexp.WordPunctTokenizer()
    for sentence in segmented_sentences:
        tokenized_sentence = word_tokenizer.tokenize(sentence.strip())
        table = str.maketrans('', '', string.punctuation)
        tokenized_sentence = [w.translate(table) for w in tokenized_sentence]
        if tokenized_sentence != '':
            processed_tweet.append(tokenized_sentence)
    
    
    if hashtag_tokens:
        for tag_token in hashtag_tokens:
            processed_tweet.append(tag_token)
    
    
    return processed_tweet


for i in x:
    x_refined.append(preprocess(i))

x_split = np.array_split(x_refined, 5)
y_split = np.array_split(y, 5)

x_test = x_split[0]
y_test = y_split[0]

x_train = np.concatenate(x_split[1:])
y_train = np.concatenate(y_split[1:])

print(x_train[0])
print(x_test[0])
print(y_train[0])
print(y_test[0])


from tqdm import tqdm
embeddings_index = {}
f = open('Code/glove.twitter.27B.200d.txt', encoding="utf8")
for line in tqdm(f):
    values = line.split()
    word = values[0]
    try:
       coefs = np.asarray(values[1:], dtype='float32')
       embeddings_index[word] = coefs
    except ValueError:
       pass
f.close()

vocab = []
count = 0
ind_to_word = {}
words____ = set()
word_list__ = []
def sent2vec(s):
    words = word_tokenize(str(s))
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
            if w not in words____:
                vocab.append(embeddings_index[w])
                word_list__.append(w)
                ind_to_word[count] = w
                count += 1
        except:
            continue
    M = np.array(M)
    
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(50)
    return v / np.sqrt((v ** 2).sum())

xtrain_glove = [sent2vec(x) for x in tqdm(x_train)]
xtest_glove = [sent2vec(x) for x in tqdm(x_test)]
xtrain_glove = np.array(xtrain_glove)
xtest_glove = np.array(xtest_glove)

logistic = linear_model.LogisticRegression()


C = np.logspace(2, 4, 10, 20)
param_grid = [
  {'penalty': ['l2'], 'solver': ['lbfgs', 'sag', 'saga' ], 'C':C, 'multi_class':['multinomial'], 'max_iter':[6000]},
 ]

clf = GridSearchCV(logistic, param_grid, cv=5, verbose=0)
best_model = clf.fit(xtrain_glove, y_train)

print("Optimal params: {}".format(clf.best_params_))
test_prediction = clf.predict(xtest_glove)

print("Accuracy: {}".format(accuracy_score(y_test,test_prediction)))
print("Precision: {}".format(precision_score(y_test,test_prediction, average=None)))
print("Recall: {}".format(recall_score(y_test,test_prediction,  average=None)))


# ------ Alternate Approach -----

x_new = []
for i in x_refined:
    l = ''
    for j in i:
        for k in j:
            l = l + ' ' + k
    x_new.append(l)
        
t = np.array(x_new)

def transformed_array(dataset):
  #x_returned = np.array()
  x_returned = np.empty((0,50), dtype='float')
  for i in dataset:
  
    l = np.array(sent2vec(i))
    #print(l.shape)
    x_returned = np.concatenate((x_returned,l.reshape(1,50)))
  return x_returned

x_returned = transformed_array(t)
x_returned = np.array(x_returned)

input_data_folds = np.array_split(x_returned, 5)
input_data_result_folds = np.array_split(y, 5)
acc_total = 0
precision_pos = 0
precision_neu = 0
precision_neg = 0

recall_pos = 0
recall_neu = 0
recall_neg = 0

for i in range(5):
    # Training data
    input_part1 = input_data_folds[:i]
    input_part2 = input_data_folds[i + 1:]
    input_train = np.concatenate(input_part1 + input_part2)
    
    input_res_part1 = input_data_result_folds[:i]
    input_res_part2 = input_data_result_folds[i + 1:]
    input_res_train = np.concatenate(input_res_part1 + input_res_part2)

    # Validation data
    input_val = input_data_folds[i]
    input_res_val = input_data_result_folds[i]
    
    # LR initialisation
    logistic = linear_model.LogisticRegression(C= 2.7825594022071245, penalty='l2', solver='liblinear')
    logistic.fit(input_train, input_res_train)

    # Predicting on validation data
    predict_val = logistic.predict(input_val)

    print("Accuracy: {}".format(accuracy_score(input_res_val,predict_val)))
    print("Precision: {}".format(precision_score(input_res_val,predict_val, average=None)))
    print("Recall: {}".format(recall_score(input_res_val,predict_val,  average=None)))
    acc_total += accuracy_score(input_res_val,predict_val)
    p  = precision_score(input_res_val,predict_val, average=None)
    precision_neg += p[0]
    precision_neu += p[1]
    precision_pos += p[2]
    r = recall_score(input_res_val,predict_val,  average=None)
    recall_neg += r[0]
    recall_neu += r[1]
    recall_pos += r[2]
        
print(acc_total/5, precision_pos/5, precision_neu/5, precision_neg/5, recall_pos/5, recall_neu/5, recall_neg/5)

def top_10(coeffs):
    for i in range(3):
        print('Taking dot')
        dot_product = v__.dot(coeffs[i])
        top10 = np.argsort(dot_product)[-50:]
        for x in top10:
            print("{}".format(word_list__[x]))
        print("--------------------------------")
            
top_10(logistic.coef_)
