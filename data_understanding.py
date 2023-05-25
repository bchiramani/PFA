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

# Load data
df = pd.read_csv('./dataset/final_dataset.csv')

# General insights
def print_insights(df):
    print("Shape : ")
    df.shape
    print("Infos : ")
    df.info()
    print("Description : ")
    df.describe()
    print("Head : ")
    df.head()

def convert_to_list(df):
    df['Dates'] = df['Dates'].apply(ast.literal_eval)
    df['Captions'] = df['Captions'].apply(ast.literal_eval)
    df['Medias'] = df['Medias'].apply(ast.literal_eval)
    df['Likes'] = df['Likes'].apply(ast.literal_eval)
    df['Comments'] = df['Comments'].apply(ast.literal_eval)
    return df

def scan_target(df):
    
    print("I-E")
    print(df['I-E'].value_counts())
    
    print("S-N")
    print(df['S-N'].value_counts())
    
    print("T-F")
    print(df['T-F'].value_counts())
    
    print("J-P")
    print(df['J-P'].value_counts())
    

def plot_post_frequency(df,title,attr):
    fig, axes = plt.subplots(ncols=4, figsize=(30, 10))
    fig.suptitle(title, fontsize=16)
    sns.scatterplot(data=df, x='ID', y=attr, hue='I-E', ax=axes[0])
    axes[0].set_title("nIntroverts and Extroverts")
    sns.scatterplot(data=df, x='ID', y=attr, hue='S-N', ax=axes[1])
    axes[1].set_title("Sensitives and Intuitives")
    sns.scatterplot(data=df, x='ID', y=attr, hue='T-F', ax=axes[2])
    axes[2].set_title("Thinkers and Feelers")
    sns.scatterplot(data=df, x='ID', y=attr, hue='J-P', ax=axes[3])
    axes[3].set_title("Judgings and Perceivings")
    plt.savefig(title+'png')

# Nb_posts
plot_post_frequency(df,'nb_posts per user','nb_posts')
df1 = df.drop(df[df['nb_posts'] > 400].index)
plot_post_frequency(df1,'nb_posts per user where it is less then 400','nb_posts')

# Length  captions
plot_post_frequency(df,'length of captions per user','mean_len')

# Hashtags
plot_post_frequency(df,'hashtags in captions per user','nb_hashtags')

# Average likes
plot_post_frequency(df,'Average likes per user','mean_likes')
df2 = df.drop(df[df['mean_likes'] > 6000].index)
plot_post_frequency(df2,'Average likes per user','mean_likes')

# Average Comments
plot_post_frequency(df,'Average comments per user','mean_comments')
df2 = df.drop(df[df['mean_comments'] > 20].index)
plot_post_frequency(df2,'Average comments per user','mean_comments')
