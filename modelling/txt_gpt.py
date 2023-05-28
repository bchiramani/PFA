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

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import TFGPT2Model, GPT2Tokenizer


df = pd.read_csv('../Dataset/trainset.csv')
df_val = pd.read_csv('../Dataset/validationset.csv')
df_test = pd.read_csv('../Dataset/testset.csv')

df=df[["user_name","Dates","text_clean","I-E","S-N","T-F","J-P"]]
df_val=df_val[["user_name","Dates","text_clean","I-E","S-N","T-F","J-P"]]
df_test=df_test[["user_name","Dates","text_clean","I-E","S-N","T-F","J-P"]]



def concatenate_captions(df):
  captions_list= []
  for user in df["text_clean"]:
    user_caption =""
    for caption_index,caption in enumerate(eval(user)):
      user_caption += caption + "  ,  "
    captions_list.append(user_caption)
  df["user_caption"]= captions_list


# Define a custom tokenizer function and call the encode_plus method of the BERT tokenizer.
def tokenize_caption(caption) :
  """
  "input_ids" contient les IDs des tokens pour chaque phrase encodée
  "attention_masks" contient des masques d'attention pour chaque phrase encodée.
  """
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  tokens = tokenizer.encode_plus(
      caption,
      add_special_tokens=True,
      max_length=128,
      padding=True,
      truncation=True,
      return_attention_mask=True
  )
  return tokens['input_ids'], tokens['attention_mask']


def tokenizing_df(df): 
  all_input_ids=[]
  all_attention_masks=[]
  for caption in df["text_clean"]:
    input_ids , attention_masks  = tokenize_caption(caption)
    all_input_ids.append(input_ids)
    all_attention_masks.append(attention_masks)

  df['input_ids_gpt2'], df['attention_masks_gpt2'] = all_input_ids, all_attention_masks
  return df


def preprocess_data(df):
    # Concatenate captions
    concatenate_captions(df)
    # Tokenize captions
    df = tokenizing_df(df)
    return df
  
# Preprocess the data
df = preprocess_data(df)
df_val = preprocess_data(df_val)
df_test = preprocess_data(df_test)

# Retrieve the necessary columns
X_train = df[['input_ids_gpt2', 'attention_masks_gpt2']]
y_train_ie = df["I-E"]
y_train_sn = df["S-N"]
y_train_tf = df["T-F"]
y_train_jp = df["J-P"]

X_val = df_val[['input_ids_gpt2', 'attention_masks_gpt2']]
y_val_ie = df_val["I-E"]
y_val_sn = df_val["S-N"]
y_val_tf = df_val["T-F"]
y_val_jp = df_val["J-P"]

X_test = df_test[['input_ids_gpt2', 'attention_masks_gpt2']]
y_test_ie = df_test["I-E"]
y_test_sn = df_test["S-N"]
y_test_tf = df_test["T-F"]
y_test_jp = df_test["J-P"]



# y_train_hot_ie = y_train_ie.copy()
# y_val_hot_ie = y_val_ie.copy()
# y_test_hot_ie = y_test_ie.copy()

# y_train_hot_sn = y_train_sn.copy()
# y_val_hot_sn = y_val_sn.copy()
# y_test_hot_sn = y_test_sn.copy()

# y_train_hot_tf = y_train_tf.copy()
# y_val_hot_tf = y_val_tf.copy()
# y_test_hot_tf = y_test_tf.copy()

# y_train_hot_jp = y_train_jp.copy()
# y_val_hot_jp = y_val_jp.copy()
# y_test_hot_jp = y_test_jp.copy()
# ohe = preprocessing.OneHotEncoder()
# y_train_enc_ie = ohe.fit_transform(np.array(y_train_ie).reshape(-1, 1)).toarray()
# y_val_enc_ie = ohe.fit_transform(np.array(y_val_ie).reshape(-1, 1)).toarray()
# y_test_enc_ie = ohe.fit_transform(np.array(y_test_ie).reshape(-1, 1)).toarray()

# y_train_enc_sn = ohe.fit_transform(np.array(y_train_sn).reshape(-1, 1)).toarray()
# y_val_enc_sn = ohe.fit_transform(np.array(y_val_sn).reshape(-1, 1)).toarray()
# y_test_enc_sn = ohe.fit_transform(np.array(y_test_sn).reshape(-1, 1)).toarray()

# y_train_enc_tf = ohe.fit_transform(np.array(y_train_tf).reshape(-1, 1)).toarray()
# y_val_enc_tf = ohe.fit_transform(np.array(y_val_tf).reshape(-1, 1)).toarray()
# y_test_enc_tf = ohe.fit_transform(np.array(y_test_tf).reshape(-1, 1)).toarray()

# y_train_enc_jp = ohe.fit_transform(np.array(y_train_jp).reshape(-1, 1)).toarray()
# y_val_enc_jp = ohe.fit_transform(np.array(y_val_jp).reshape(-1, 1)).toarray()
# y_test_enc_jp = ohe.fit_transform(np.array(y_test_jp).reshape(-1, 1)).toarray()


train_input_ids_ragged = tf.ragged.constant(X_train['input_ids_gpt2'].tolist())
train_attention_mask_ragged = tf.ragged.constant(X_train['attention_masks_gpt2'].tolist())

val_input_ids_ragged = tf.ragged.constant(X_val['input_ids_gpt2'].tolist())
val_attention_mask_ragged = tf.ragged.constant(X_val['attention_masks_gpt2'].tolist())

test_input_ids_ragged = tf.ragged.constant(X_test['input_ids_gpt2'].tolist())
test_attention_mask_ragged = tf.ragged.constant(X_test['attention_masks_gpt2'].tolist())

#  FINAL
# convert the ragged tensors to dense tensors
train_input_ids = train_input_ids_ragged.to_tensor()
train_attention_mask = train_attention_mask_ragged.to_tensor()

val_input_ids = val_input_ids_ragged.to_tensor()
val_attention_mask = val_attention_mask_ragged.to_tensor()

test_input_ids = test_input_ids_ragged.to_tensor()
test_atention_mask = test_attention_mask_ragged.to_tensor()


MAX_LEN=128
def create_model(gpt2_model, max_len=MAX_LEN): 
  #Define the input layers
  input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32)
  attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32)

  # Pass the inputs through the GPT-2 model
  outputs = gpt2_model({'input_ids': input_ids, 'attention_mask': attention_mask})

  # Get the last hidden state output from the GPT-2 model
  last_hidden_state = outputs[0]

  # Add a dense layer for classification
  dense_layer = tf.keras.layers.Dense(1, activation='sigmoid')(last_hidden_state[:, -1, :])

  # Define the model
  model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=dense_layer)

  # Compile the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model 



def train_test_function (model, title,train_input_ids,train_attention_mask,y_train_ie,val_input_ids,val_attention_mask,y_val_ie,test_input_ids,test_atention_mask,y_test_ie):
    history_gpt2_ie = model.fit(
    [train_input_ids, train_attention_mask], y_train_ie, 

    validation_data=([val_input_ids, val_attention_mask], y_val_ie),

    epochs=5, batch_size=32)
    y_pred_ie = model.predict([test_input_ids, test_atention_mask])
    predictions_ie = np.round(y_pred_ie).astype(int)
    with open('gpt_results.txt', 'a') as file:
        file.write('\tClassification Report for GPT2 For '+title+':\n\n',classification_report(y_test_ie,predictions_ie, target_names=['0', '1']))

    # print('\tClassification Report for GPT2 For '+title+':\n\n',classification_report(y_test_ie,predictions_ie, target_names=['0', '1']))
    cm = confusion_matrix(y_test_ie, predictions_ie)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

gpt2_model = TFGPT2Model.from_pretrained('gpt2', output_hidden_states=True)
model = create_model(gpt2_model, MAX_LEN)
model.summary()   
train_test_function (model, "Ii-E",train_input_ids,train_attention_mask,y_train_ie,val_input_ids,val_attention_mask,y_val_ie,test_input_ids,test_atention_mask,y_test_ie)

    
    
    
    
    
    
    
    
# history_gpt2_ie = model.fit(
# [train_input_ids, train_attention_mask], y_train_ie, 

# validation_data=([val_input_ids, val_attention_mask], y_val),

# epochs=5, batch_size=32)
# y_pred_ie = model.predict([test_input_ids, test_atention_mask])
# predictions_ie = np.round(y_pred_ie).astype(int)

# print('\tClassification Report for GPT2 For I-E:\n\n',classification_report(y_test_ie,predictions_ie, target_names=['0', '1']))
# cm = confusion_matrix(y_test_ie, predictions_ie)
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# history_gpt2_sn = model.fit(
#     [train_input_ids, train_attention_mask], y_train_sn, 

#     validation_data=([val_input_ids, val_attention_mask], y_val_sn),

#     epochs=5, batch_size=32)
# y_pred_sn = model.predict([test_input_ids, test_atention_mask])
# predictions_sn = np.round(y_pred_sn).astype(int)
# print('\tClassification Report for GPT2 FOR S-N :\n\n',classification_report(y_test_sn,predictions_sn, target_names=['0', '1']))

# cm = confusion_matrix(y_test_sn, predictions_sn)
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()
 
# history_gpt2_tf = model.fit(
#     [train_input_ids, train_attention_mask], y_train_tf, 

#     validation_data=([val_input_ids, val_attention_mask], y_val_tf),

#     epochs=5, batch_size=32)
# y_pred_tf = model.predict([test_input_ids, test_atention_mask])
# predictions_tf = np.round(y_pred_tf).astype(int)
# print('\tClassification Report for GPT2 FOR T-F :\n\n',classification_report(y_test_tf,predictions_tf, target_names=['0', '1']))
# cm = confusion_matrix(y_test_tf, predictions_tf)
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()



# history_gpt2_jp = model.fit(
#     [train_input_ids, train_attention_mask], y_train_jp, 

#     validation_data=([val_input_ids, val_attention_mask], y_val_jp),

#     epochs=5, batch_size=32)
# y_pred_jp = model.predict([test_input_ids, test_atention_mask])
# predictions_jp = np.round(y_pred_jp).astype(int)
# print('\tClassification Report for GPT2 FOR J-P :\n\n',classification_report(y_test_jp,predictions_jp, target_names=['0', '1']))
# cm = confusion_matrix(y_test_jp, predictions_jp)
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()