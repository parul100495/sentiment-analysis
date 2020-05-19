"""
@pargupta
02/15/2020
Sentiment Analysis using Logistic Regression

"""

import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import os
import re
import nltk
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('sentiwordnet')



stopwords = set(stopwords.words('english'))

from nltk.tokenize import sent_tokenize, word_tokenize

dictionary = set(nltk.corpus.words.words())
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer()


readMe = open('Code/DataSubmission/annotation2.csv','r', encoding = "cp1252").readlines() 
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
        if lemmatize(first).lower() in dictionary: 
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

total_train = {}

for x_ in x_train:
    for segment in x_:
        for token in segment:
            total_train[token] = total_train.get(token,0) + 1

def convert_to_feature(tweets,remove_stop_words,n): 
    feature_dicts = []
    for tweet in tweets:
        feature_dict = {}
        if remove_stop_words:
            for segment in tweet:
                for token in segment:
                    try:
                        if token not in stopwords and (n<=0 or total_train[token]>=n):
                            feature_dict[token] = feature_dict.get(token,0) + 1
                    except Exception as e:
                        pass
        else:
            for segment in tweet:
                for token in segment:
                    if n<=0 or total_train[token]>=n:
                        feature_dict[token] = feature_dict.get(token,0) + 1
        feature_dicts.append(feature_dict)
    return feature_dicts


train_set = convert_to_feature(x_train,True,2)
test_set = convert_to_feature(x_test,True,2)

training_data = vectorizer.fit_transform(train_set)
testing_data = vectorizer.transform(test_set)

logistic = linear_model.LogisticRegression()

C = np.logspace(0, 4, 10)
param_grid = [
  {'penalty': ['l1'], 'solver': [ 'liblinear'], 'C': C},
  {'penalty': ['l2'], 'solver': ['liblinear'], 'C':C},
  {'penalty': ['l2'], 'solver': ['newton-cg'], 'C':C, 'multi_class':['multinomial']},
 ]


clf = GridSearchCV(logistic, param_grid, cv=5, verbose=0)

best_model = clf.fit(training_data, y_train)

print("Optimal params: {}".format(clf.best_params_))

test_prediction = clf.predict(testing_data)

print("Accuracy: {}".format(accuracy_score(y_test,test_prediction)))
print("Precision: {}".format(precision_score(y_test,test_prediction, average=None)))
print("Recall: {}".format(recall_score(y_test,test_prediction,  average=None)))

#------- Alternate Approach -------------
# Create grid search using 5-fold cross validation
input_data_folds = np.array_split(train_set, 5)
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
    training_data = vectorizer.fit_transform(input_train)
    validation_data = vectorizer.transform(input_val)

    # Training LR
    logistic.fit(training_data, input_res_train)

    # Predicting on validation data
    predict_val = logistic.predict(validation_data)

    # print("Accuracy: {}".format(accuracy_score(input_res_val,predict_val)))
    # print("Precision: {}".format(precision_score(input_res_val,predict_val, average=None)))
    # print("Recall: {}".format(recall_score(input_res_val,predict_val,  average=None)))
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



def print_top_feats(vectorizer, clf, class_labels=clf.classes_):
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top_ = np.argsort(clf.coef_[i])[-10:]
        print("{}: {}".format(class_label," ".join(feature_names[j] for j in top_)))

print_top_feats(vectorizer, clf.best_estimator_)