import nltk 
import re 
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nltk.download('words')
nltk.download('punkt')

from sqlalchemy import create_engine
from nltk.corpus import words, stopwords
from nltk.tokenize import word_tokenize

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def check_english_word(word, english_words):
    if word in english_words:
        return True
    else:
        return False
    
def check_english_sent(sent, english_words):
    results = []
    for token in word_tokenize(sent.lower()):
        if len(token) > 1:
            results.append(check_english_word(token, english_words))
    return any(results)


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = [column.split('-')[0] for column in row.values]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'], axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join='inner', axis=1)

    # drop duplicated rows
    df.drop_duplicates(inplace=True)
    
    # drop rows with duplicate ids
    # loop over duplicated ids
    for id_ in df.id[df.id.duplicated()].unique():
        # extract rows with id_
        df_id = df[df.id == id_]

        # extract first 4 columns
        df_id_first_columns = df_id.iloc[0, :4]

        # extract category columns
        df_id_categories = df_id.iloc[:, 4:]

        # select maximum category in each column over all rows
        df_id_categories_max = df_id_categories.max(axis=0)

        # merge new categories with first 4 columns
        df_id = pd.concat([df_id_first_columns, df_id_categories_max], axis=0, ignore_index=True)

        # change df_id index to df column names
        df_id.index = df.columns

        # remove old id_ rows in df
        df = df[df.id != id_]

        # append new id_ row
        df = df.append(df_id, ignore_index=True)
        
    # drop non-english messages
    english_words = set(words.words())    
    is_english = df['message'].apply(lambda x: check_english_sent(x, english_words))
    df = df[is_english]
    
    # replace URLs
    url_regex = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    df['message'] = df['message'].str.replace(url_regex, 'urlplaceholder', regex=True)
    
    # remove bit.ly URLS
    bitly_regex = r'(?:http.)*(?:www.)*bit.ly .*\s{,1}'
    df['message'] = df['message'].str.replace(bitly_regex, 'urlplaceholder', regex=True)
    
    return df

def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('clean', engine, if_exists='replace', index=False)      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()