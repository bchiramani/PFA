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
df['nb_posts'] = [len(c) for c in df['Dates']]

df['mean_len'] = df['len_caption'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)

df['nb_hashtags'] = df['Captions'].apply(lambda x: sum([caption.count('#') for caption in x]))

"""
Calculate average number of likes per person.
"""
df['mean_likes'] = df['Likes'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)


"""
Calculate average number of comments  per person.
"""
df['mean_comments'] = df['Comments'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)


# Define a lambda function to check if the length of the list of posts is zero
is_empty = lambda x: len(x) == 0

df['empty_posts'] = df['Dates'].apply(is_empty)
users= df.loc[df['empty_posts'], 'user_name'].tolist()

print("Number of users with empty list of posts: ",len(users))
print("those folders are empty : ")
if len(users) > 0:
    print("The following users have an empty list of posts: \n")
    print("\n".join(users))
