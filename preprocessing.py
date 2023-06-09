import pandas as pd
import numpy as np
import os
import csv
import pandas as pd
import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import re
import ast

df = pd.read_csv('./dataset/final_dataset.csv')
df = df.rename(columns={'Unnamed: 0': 'ID'})

def add_features(df):
    df['nb_posts'] = [len(c) for c in df['Dates']]

    df['mean_len'] = df['len_caption'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)

    df['nb_hashtags'] = df['Captions'].apply(lambda x: sum([caption.count('#') for caption in x]))

    df['mean_likes'] = df['Likes'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)

    df['mean_comments'] = df['Comments'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)

def log_users_with_zero_post(df):
    is_empty = lambda x: len(x) == 0

    df['empty_posts'] = df['Dates'].apply(is_empty)
    users= df.loc[df['empty_posts'], 'user_name'].tolist()
    print("Number of users with empty list of posts: ",len(users))
    print("those folders are empty : ")
    
    if len(users) > 0:
        print("The following users have an empty list of posts: \n")
        print("\n".join(users))
        
#Clean emojis from text
def strip_emoji(text):
  """
  Remove emojis from a string.
  """
  emoji_pattern = emoji_pattern = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r'', text)
#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text
#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(caption):
    new_caption = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', caption)) #remove last hashtags
    new_caption2 = " ".join(word.strip() for word in re.split('#|_', new_caption)) #remove hashtags symbol from words in the middle of the sentence
    return new_caption2

#Filter special characters such as & and $ present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)
def remove_mult_spaces(text): # remove multiple spaces
    return re.sub("\s\s+" , " ", text)

def clean(df):
  texts_new = []
  for caption in df["user_caption"]:
      texts_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(strip_emoji(caption))))))
  df['text_clean'] = texts_new
def encoding (df , att):
  df[att] = df[att].map({1.0:1,0.0:0})
  
log_users_with_zero_post(df)
add_features(df)
clean(df)
encoding(df,'I-E')
encoding(df,'S-N')
encoding(df,'T-F')
encoding(df,'J-P')
df.to_csv("preprocessed_dataset.csv")