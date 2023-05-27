import pandas as pd
import numpy as np
import os
import csv
import pandas as pd
import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
import ast
import emoji 
df = pd.read_csv('./dataset/final_dataset.csv')
df = df.rename(columns={'Unnamed: 0': 'ID'})

def add_features(df):
    df['nb_posts'] = [len(c) for c in df['Dates']]
    
    df['len_caption'] = [[len(x) for x in c] for c in df['Captions']]
    
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
    if isinstance(text, str):
        emoji_pattern = emoji.get_emoji_regexp()
        return emoji_pattern.sub(r'', text)
    return text
#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text): 
    if isinstance(text, str):
        text = text.replace('\r', '').replace('\n', ' ').lower()
        text = re.sub(r"(?:\@|https?\://)\S+", "", text)
        text = re.sub(r'[^\x00-\x7f]', r'', text)
        banned_list = string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
        table = str.maketrans('', '', banned_list)
        text = text.translate(table)
    return text
#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(caption):
    if isinstance(caption, str):
        new_caption = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', caption))
        new_caption2 = " ".join(word.strip() for word in re.split('#|_', new_caption))
        return new_caption2
    return caption

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
    if isinstance(text, str):
        return re.sub("\s\s+", " ", text)
    return text

def clean(df):
    texts_new = []
    for captions in df["Captions"]:
        cleaned_captions = []
        for caption in captions:
            clean_caption = strip_emoji(caption)
            clean_caption = strip_all_entities(clean_caption)
            clean_caption = clean_hashtags(clean_caption)
            clean_caption = filter_chars(clean_caption)
            clean_caption = remove_mult_spaces(clean_caption)
            cleaned_captions.append(clean_caption)
        texts_new.append(cleaned_captions)
    df['text_clean'] = texts_new
  
def encoding (df , att):
  df[att] = df[att].map({1.0:1,0.0:0})
  
log_users_with_zero_post(df)
df['Dates'] = df['Dates'].apply(ast.literal_eval)
df['Captions'] = df['Captions'].apply(ast.literal_eval)
df['Medias'] = df['Medias'].apply(ast.literal_eval)
df['Likes'] = df['Likes'].apply(ast.literal_eval)
df['Comments'] = df['Comments'].apply(ast.literal_eval)
add_features(df)
clean(df)
encoding(df,'I-E')
encoding(df,'S-N')
encoding(df,'T-F')
encoding(df,'J-P')
df.to_csv("./Dataset/preprocessed_dataset.csv")