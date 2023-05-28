import pandas as pd
import numpy as np
import os
import csv
import tensorflow as tf

import emoji
import re, string
import nltk

import matplotlib.pyplot as plt
import seaborn as sns

#transformers
from transformers import BertTokenizerFast
from transformers import TFBertModel

from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

#Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ------- Class Balancing by RandomOverSampler -----------------

# It is a technique used for balancing the classes in an 
# imbalanced dataset. In an imbalanced dataset, one class 
# has significantly more examples than the other class(es), 
# which can lead to poor performance of machine learning 
# models, especially for the minority class. RandomOverSampler 
# works by randomly oversampling the minority class examples 
# to balance the class distribution. This is done by creating 
# synthetic examples of the minority class by randomly 
# duplicating existing examples. The oversampling is 
# performed until the number of examples in the minority 
# class is equal to the number of examples in the majority 
# class.

def balancing_function(df):
    ros = RandomOverSampler()
    train_x, train_y = ros.fit_resample(np.array(df[['text_clean', 'I-E', 'S-N', 'T-F', 'J-P']]), np.array(df['I-E']))
    train_os = pd.DataFrame(train_x, columns=['text_clean', 'I-E', 'S-N', 'T-F', 'J-P'])
    train_os['I-E'] = train_y
    return train_os

def get_x_y(train_os):
    X_train = train_os['text_clean'].values
    y_train_ie = train_os["I-E"].values
    y_train_ie = y_train_ie.astype(int)
    y_train_sn = train_os["S-N"].values
    y_train_sn = y_train_sn.astype(int)
    y_train_tf = train_os["T-F"].values
    y_train_tf = y_train_tf.astype(int)
    y_train_jp = train_os["J-P"].values
    y_train_jp = y_train_jp.astype(int)
    return X_train,y_train_ie,y_train_sn , y_train_tf,y_train_jp
def tokenize(X_train,X_test):
    #Tokenize captions
    clf_ie = CountVectorizer()
    X_train_cv =  clf_ie.fit_transform(X_train)
    X_test_cv = clf_ie.transform(X_test)

    # Create the TF-IDF (term-frequency times inverse document-frequency) versions of the tokenized captions
    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
    X_train_tf = tf_transformer.transform(X_train_cv)
    X_test_tf = tf_transformer.transform(X_test_cv)
    return X_train_tf, X_test_tf

def model(X_train,y_train, X_test,y_test,title):
    
    # Define the Naive Bayes Classifier model
    nb_clf = MultinomialNB()

    nb_clf.fit(X_train, y_train)

    nb_pred = nb_clf.predict(X_test)
    with open('naive_bayes_results.txt', 'a') as f:
        f.write('\tClassification Report for Naive Bayes for '+title+':\n\n')
        f.write(classification_report(y_test, nb_pred, target_names=['0', '1']))
    


df = pd.read_csv('../Dataset/trainset.csv')
df_val = pd.read_csv('../Dataset/validationset.csv')
df_test = pd.read_csv('../Dataset/testset.csv')


df=df[["user_name","Dates","text_clean","I-E","S-N","T-F","J-P"]]
df_val=df_val[["user_name","Dates","text_clean","I-E","S-N","T-F","J-P"]]
df_test=df_test[["user_name","Dates","text_clean","I-E","S-N","T-F","J-P"]]

train_os = balancing_function(df)

X_train,y_train_ie,y_train_sn , y_train_tf,y_train_jp= get_x_y(train_os)
X_val = df_val["text_clean"].values
y_val_ie = df_val["I-E"].values
y_val_sn = df_val["S-N"].values
y_val_tf = df_val["T-F"].values
y_val_jp = df_val["J-P"].values

X_test = df_test["text_clean"].values
y_test_ie = df_test["I-E"].values
y_test_sn = df_test["S-N"].values
y_test_tf = df_test["T-F"].values
y_test_jp = df_test["J-P"].values


# --------Baseline model : Naive Bayes Classifier --------------------
# Before implementing BERT, we will define a simple Naive Bayes baseline as a baseline

X_train_tok, X_test_tok = tokenize(X_train,X_test)
model(X_train_tok,y_train_ie, X_test_tok,y_test_ie,"I-E")
model(X_train_tok,y_train_sn, X_test_tok,y_test_sn,"S-N")
model(X_train_tok,y_train_tf, X_test_tok,y_test_tf,"T-F")
model(X_train_tok,y_train_jp, X_test_tok,y_test_jp,"J-P")