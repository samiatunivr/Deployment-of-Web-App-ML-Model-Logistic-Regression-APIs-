# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk  # natural language toolkit for text manipulation
import re  # file handling
from nltk.corpus import stopwords  # we need to use list of words ltr


train_data = pd.read_csv(r"C:\Users\Sami\Desktop\genre\input\data\train.csv")


def clean_synopsis(synopsis_text):
    """ First we remove backslahes
    then remove everything except text,
    then white spaces and change all
    letter cases to lowercase """

    synopsis_text = re.sub("\'", "", synopsis_text)
    synopsis_text = re.sub("[^a-zA-Z]", " ", synopsis_text)
    synopsis_text = ' '.join(synopsis_text.split())
    synopsis_text = synopsis_text.lower()

    return synopsis_text


def stopword_removal(synopsis_text):
    """ We use the English stop word (thanks to nltk)
     with the help of list comprehension to remove all the stop
    words from the movie synopsis, we then remove
    any white space and return a list of cleaned synopsis"""

    stop_words = set(stopwords.words("english"))
    synopsis_text = [t for t in synopsis_text.split() if t not in stop_words]
    synopsis_text = ' '.join(synopsis_text)

    return synopsis_text


def splitString(textlist):

    splitedtext = textlist.split(' ')
    return splitedtext

# we create a function that takes care of synopsis of the movie
def synopsis_analysis(movie_data):
    """ First we remove backslah,
    lowercase the letters, etc.  second
    we remove stop words from the synopsis,
     we use lambda function
     input: a data frame with synopsis column
     output: clean data"""

    movie_data_new = movie_data
    movie_data_new['synopsis_clean'] = movie_data_new['synopsis'].apply(lambda x: clean_synopsis(x))
    movie_data_new['synopsis_clean'] = movie_data_new['synopsis_clean'].apply(lambda x: stopword_removal(x))

    if 'genres' in movie_data_new.columns:

        movie_data_new['new_genre'] = movie_data_new['genres'].apply(lambda x: splitString(x))

    return movie_data_new

