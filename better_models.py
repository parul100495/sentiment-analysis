"""
@pargupta
02/17/2020
Sentiment Analysis using Naive Bayes / ELMO+LR / Decision tree

"""
import pandas as pd
import numpy as np
import csv
import nltk
import string
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import random

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

import pickle
import re
from collections import Counter
from string import punctuation
import tweepy
from tweepy import OAuthHandler
import json
from wordcloud import WordCloud

from IPython.display import IFrame
import folium
from folium import plugins
from folium.plugins import MarkerCluster, FastMarkerCluster, HeatMapWithTime

pd.set_option('display.max_colwidth', -1)
plt.style.use('seaborn-white')

readMe = open('Code/DataSubmission/annotation2.csv','r', encoding = "cp1252").readlines() 
x = []
y = []

for i in range(1,251):
    r = readMe[i].split(',')
    x.append(r[0])
    y.append(r[1])

def processTweet(tweet):
    tweet = re.sub(r'\&\w*;', '', tweet)

    tweet = re.sub('@[^\s]+','',tweet)

    tweet = re.sub(r'\$\w*', '', tweet)

    tweet = tweet.lower()

    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)

    tweet = re.sub(r'#\w*', '', tweet)

    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)

    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)

    tweet = re.sub(r'\s\s+', ' ', tweet)

    tweet = tweet.lstrip(' ') 

    tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
    return tweet

x_refined = []
for i in x:
    x_refined.append(processTweet(i))


nltk.download('stopwords')
stop_words = stopwords.words('english')

def text_process(raw_text):
    nopunc = [char for char in list(raw_text) if char not in string.punctuation]

    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.lower().split() if word.lower() not in stopwords.words('english')]

import tensorflow_hub as hub
import tensorflow as tf

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)


def elmo_vectors(x):
  embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))

x_split = np.array_split(elmo_train, 5)
y_split = np.array_split(y, 5)

x_test = x_split[0]
y_test = y_split[0]

x_train = np.concatenate(x_split[1:])
y_train = np.concatenate(y_split[1:])

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV


logistic = linear_model.LogisticRegression()

C = np.logspace(2,4,10,20)
param_grid = [ {'penalty': ['l2'], 'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'], 'C':C, 'multi_class':['multinomial'], 'max_iter': [6000]},
 ]

clf = GridSearchCV(logistic, param_grid, cv=5, verbose=0)

best_model = clf.fit(x_train, y_train)

print("Optimal parameters for LR: {}".format(clf.best_params_)) #To print out the best discovered combination of the parameters

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
test_prediction = clf.predict(x_test)
print("Accuracy: {}".format(accuracy_score(y_test,test_prediction)))
print("Precision: {}".format(precision_score(y_test,test_prediction, average=None)))
print("Recall: {}".format(recall_score(y_test,test_prediction,  average=None)))

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
pipeline = Pipeline([
    ('bow', CountVectorizer(strip_accents='ascii',
                            stop_words='english',
                            lowercase=True)),  
    ('tfidf', TfidfTransformer()), 
    ('classifier', MultinomialNB()), 
])


parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'classifier__alpha': (1e-2, 1e-3),
             }
             
grid = GridSearchCV(pipeline, cv=5, param_grid=parameters, verbose=1)
grid.fit(X_train,y_train)

print("Optimal params: {}".format(grid.best_params_))

test_prediction = grid.predict(x_test_)

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
print('accuracy score: {}'.format(accuracy_score(y_test_, test_prediction)))

print("Precision: {}".format(precision_score(y_test_,test_prediction, average=None)))
print("Recall: {}".format(recall_score(y_test_,test_prediction,  average=None)))

from sklearn.tree import DecisionTreeClassifier

pipeline = Pipeline([
    ('bow', CountVectorizer(strip_accents='ascii',
                            stop_words='english',
                            lowercase=True)), 
    ('tfidf', TfidfTransformer()),  
    ('classifier', DecisionTreeClassifier()),  
])

parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'classifier__min_samples_split': [2,4,6,8],
              'classifier__criterion': ['gini', 'entropy'],
              'classifier__splitter': ['best', 'random'],
              'classifier__max_depth': [2,3,4,5,6,7,8,9]
             }

grid = GridSearchCV(pipeline, cv=5, param_grid=parameters, verbose=1)
grid.fit(X_train,y_train)

print("Optimal params: {}".format(grid.best_params_))
test_prediction = grid.predict(x_test_)

print('accuracy score: {}'.format(accuracy_score(y_test_, test_prediction)))

print("Precision: {}".format(precision_score(y_test_,test_prediction, average=None)))
print("Recall: {}".format(recall_score(y_test_,test_prediction,  average=None)))

# ---- Alternate approach -----

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn import linear_model

# Create grid search using 5-fold cross validation
input_data_folds = np.array_split(elmo_train, 5)
input_data_result_folds = np.array_split(y, 5)
# clf = GridSearchCV(logistic, param_grid, cv=5, verbose=0)
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
    # print(input_train[0])
    # training_data = transformed_array(input_train)
    # validation_data = [sent2vec(x) for x in tqdm(input_val)]
    # tdata = np.array(training_data)
    # vdata = np.array(validation_data)
    # print(tdata.shape)
    # Training LR
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