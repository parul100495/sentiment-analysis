'''
@pargupta

Please following instructions in readme to execute this script.

'''
import pandas as pd
import numpy as np
import sklearn
import os
import xml.etree.ElementTree as ET
import re
import shutil
from collections import Counter
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
%matplotlib inline

# To read xml file review sample
read_me = open('sorted_data/apparel/all.review','r', encoding = "ISO-8859-1").readlines() #read = r
for i in range(50):
    print(read_me[i])

negative_dict = {}
positive_dict = {}

def process_xml(list_reviews):
    pos_count = 0
    neg_count = 0
    rev = []
    label = False
    rev_add = True

    for i in range(len(list_reviews)):
        if list_reviews[i] != '</review>\n':
            if  list_reviews[i] == '<asin>\n':
                rev.append('asin/'+list_reviews[i+1])    
            if  list_reviews[i] == '<product_name>\n':
                rev.append('product_name/'+list_reviews[i+1])
            if list_reviews[i] == '<rating>\n':
                if float(list_reviews[i+1]) > 3:
                  label = True
                elif float(list_reviews[i+1]) < 3:
                  label = False
                else:
                  rev_add = False
            if list_reviews[i] == '<review_text>\n':
                review_text = list_reviews[i+1].lower()
                review_text = re.sub(r'[^\w\s]','',review_text)
                review_text = ''.join([i for i in review_text if not i.isdigit()])
                rev.append('review_text/'+review_text)
        elif list_reviews[i] == '</review>\n':
            if rev_add:
                if label:
                    pos_count = pos_count + 1
                    r = 'review'+ str(pos_count)
                    positive_dict[r] = rev
                else:
                    neg_count = neg_count + 1
                    r = 'review'+ str(neg_count)
                    negative_dict[r] = rev
            rev_add = True
            rev = []

all_rev = open('sorted_data/apparel/all.review','r', encoding = "ISO-8859-1").readlines()
process_xml(all_rev)

def convert_to_df(d):
    df = pd.DataFrame(columns=['asin','product_name','review_text'])
    count = 0
    for i,k in d.items():
        df.loc[count] = [k[0].split("/")[1].split("\n")[0],
                         k[1].split("/")[1].split("\n")[0],
                         k[2].split("/")[1].split("\n")[0]]
        count = count + 1
    return df

negative_df = convert_to_df(negative_dict)
positive_df = convert_to_df(positive_dict)

positive_df['Class'] = "pos"
negative_df['Class'] = "neg"

all_rev_df = pd.concat([positive_df, negative_df])

from nltk.tokenize import word_tokenize

def tokenize_words(w):
    return word_tokenize(w)

def remove_irrelevent_words(word_list):
    list_words = []
    for m in word_list:
        l = [item for item in m]
        list_words.append(l)
    word_list = list_words
    #print(len(word_list))
    final_word_list = []
    for m in word_list:
        final_review_text = []
        for w in m:
            #print(w)
            if len(w) > 3 and w != 'this' and w != 'there' and w != 'they':
                
                final_review_text.append(w)
        final_word_list.append(final_review_text)
    return final_word_list

def remove_irrelevent_word(word):
    final_review_text = []
    for w in m:
        #print(w)
        if len(w) > 3 and w != 'this' and w != 'there' and w != 'they':
            final_review_text.append(w)
    return final_review_text

from nltk.probability import FreqDist
def bag_of_words_from_list(word_list):
    all_words = []
    for i in word_list:
        for w in i:
            all_words.append(w.lower())
    all_words = FreqDist(all_words)
    return all_words

word_list =  [tokenize_words(i) for i in list(all_rev_df['review_text'])]
word_list = remove_irrelevent_words(word_list)
all_words = bag_of_words_from_list(word_list)

def get_all_words(word_list):
    list_of_all_words = []
    for i in word_list:
        for w in i:
            list_of_all_words.append(w)
    return list_of_all_words

from wordcloud import WordCloud
import matplotlib.pyplot as plt
list_of_all_words = get_all_words(word_list)
wordcloud = WordCloud(max_font_size=40).generate(' '.join(list_of_all_words))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

import random
def create_doc(rev, word_list):
    class_list = list(rev['Class'])
    docs =  []
    for i in range(len(word_list)):
        docs.append((word_list[i], class_list[i]))
    
    random.shuffle(docs)
    return docs

docs = create_doc(all_rev_df, word_list)

def get_features(docs, all_words, num_of_words):
    most_common = []    
    for i in all_words.most_common(num_of_words):
        most_common.append(i[0])

    word_feat = most_common
    words = tokenize_words(docs)
    features = {}
    for w in word_feat:
        features[w] = (w in words)
    return features

def get_set_of_features(docs, all_words, num_of_words):
    set_of_features = []
    for rev, cl in docs:
        feat = get_features(' '.join(rev), all_words, num_of_words)
        set_of_features.append((feat, cl))
    return set_of_features

from sklearn.model_selection import train_test_split
train, test = train_test_split(docs, test_size=0.3, random_state=50)

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

def naive_bayes(train, test, number_of_words):
    set_of_features_train = get_set_of_features(train, all_words, number_of_words)
    set_of_features_test = get_set_of_features(test, all_words, number_of_words)
    
    classifier = NaiveBayesClassifier.train(set_of_features_train)
    acc = nltk.classify.accuracy(classifier, set_of_features_test) * 100

    return classifier, most_info_feat, acc

# best classifier hypertuning finding
best_acc = 0
best_classifier = None
best_i = 0
dict_by_number_of_words = {}
for i in range(100, 30001, 500):
    classifier, most_info_feat, acc = naive_bayes(train, test, i)
    dict_by_number_of_words[i] = acc
    if acc > best_acc:
        best_acc = acc
        best_classifier = classifier
        best_i = i

# best at 2000
classifier = best_classifier
classifier.show_most_informative_features(50)

# graph for hypertuning
plt.plot(list(dict_by_number_of_words.keys()), list(dict_by_number_of_words.values()), color="blue", label='Train accuracy',marker='')
plt.title('Accuracy vs number of common words in feature set')
plt.xlabel('Number of common words in feature set')
plt.ylabel('Accuracy (in %)')
#plt.legend(loc='best')
plt.show()

def naive_bayes_classify(classifier, test):
    set_of_features_test = get_set_of_features(test, all_words, 2000)
    acc = nltk.classify.accuracy(classifier, set_of_features_test) * 100

    return acc

def process_xml_cat(list_reviews):
    pos_count = 0
    neg_count = 0
    rev = []
    label = False
    rev_add = True

    for i in range(len(list_reviews)):
        if list_reviews[i] != '</review>\n':
            if  list_reviews[i] == '<asin>\n':
                rev.append('asin/'+list_reviews[i+1])    
            if  list_reviews[i] == '<product_name>\n':
                rev.append('product_name/'+list_reviews[i+1])
            if list_reviews[i] == '<rating>\n':
                if float(list_reviews[i+1]) > 3:
                  label = True
                elif float(list_reviews[i+1]) < 3:
                  label = False
                else:
                  rev_add = False
            if list_reviews[i] == '<review_text>\n':
                review_text = list_reviews[i+1].lower()
                review_text = re.sub(r'[^\w\s]','',review_text)
                review_text = ''.join([i for i in review_text if not i.isdigit()])
                rev.append('review_text/'+review_text)
        elif list_reviews[i] == '</review>\n':
            if rev_add:
                if label:
                    pos_count = pos_count + 1
                    r = 'review'+ str(pos_count)
                    positive_dict_a[r] = rev
                else:
                    neg_count = neg_count + 1
                    r = 'review'+ str(neg_count)
                    negative_dict_a[r] = rev
            rev_add = True
            rev = []

cat = ['automotive', 'baby', 'beauty', 'camera_&_photo', 'cell_phones_&_service', 'computer_&_video_games', 'grocery', 'health_&_personal_care', 'magazines', 'office_products']
dict_by_cat = {}
for i in cat:
    positive_dict_a = {}
    negative_dict_a = {}
    all_rev_cat = open('sorted_data/' + i + '/all.review','r', encoding = "ISO-8859-1").readlines()
    process_xml_cat(all_rev_cat)
    cat_negative_df = convert_to_df(negative_dict_a)
    cat_positive_df = convert_to_df(positive_dict_a)
    cat_positive_df['Class'] = "pos"
    cat_negative_df['Class'] = "neg"
    all_rev_df_cat = pd.concat([cat_positive_df, cat_negative_df])
    word_list =  [tokenize_words(i) for i in list(all_rev_df_cat['review_text'])]
    docs_cat = create_doc(all_rev_df_cat, word_list)
    acc = naive_bayes_classify(classifier, docs_cat)
    dict_by_cat[i] = acc

# graph on multiple categories
x = [1,2,3,4,5,6,7,8,9,10]
labels = ['automotive', 'baby', 'beauty', 'camera_&_photo', 'cell_phones_&_service', 'computer_&_video_games', 'grocery', 'health_&_personal_care', 'magazines', 'office_products']
plt.bar(x, list(dict_by_cat.values()), label='Test accuracy')
plt.title('Accuracy vs category')
plt.xticks(x, labels, rotation='vertical')
plt.xlabel('Category')
plt.ylabel('Accuracy (in %)')
plt.show()
