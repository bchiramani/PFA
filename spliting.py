import cv2
import os
import pandas as pd
import os
import cv2
import csv
import pandas as pd
import datetime
import json
import lzma
from sklearn.model_selection import train_test_split



dataset = pd.read_csv('./dataset/final_dataset.csv')
train_path = "./dataset/trainset.csv"
validation_path = "./dataset/valset.csv"
test_path ="./dataset/testset.csv"


def split (dataset , train_path, validation_path,test_path):
    X = dataset[["user_name", "Dates", "Captions", "Medias", "Likes", "Comments"]]
    y = dataset[["I-E", "S-N", "T-F", "J-P"]]

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Further splitting the test set into validation and testing sets
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(train_path)
    val_df.to_csv(validation_path)
    test_df.to_csv(test_path)


split (dataset , train_path, validation_path,test_path)