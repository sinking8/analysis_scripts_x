import string
import pandas as pd
import numpy as np
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from collections import Counter

# Warnings Filter
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def pretext_pipeline(df):
    # Drop NaN values in text column
    df = df.dropna(subset=['text'])

    return df

# Calculate Top Hashtags
def top_hashtags(df,curr_counter=Counter(),top_common=10):
    df = pretext_pipeline(df)
    df['hashtags'] = df['text'].apply(lambda x: re.findall(r"#\w+", x))
    all_hashtags = [hashtag for sublist in df['hashtags'] for hashtag in sublist]
    counter = Counter(all_hashtags)
    counter_1 = counter+curr_counter
    top_hashtags = Counter(dict(counter_1.most_common(top_common)))
    
    del counter_1
    del counter
    
    return top_hashtags

# Calculate Top Urls
def top_urls(df, curr_counter=Counter(), top_common=10):
    # Extract URLs using a regex pattern
    df = pretext_pipeline(df)
    df['urls'] = df['text'].apply(lambda x: re.findall(r'https?://\S+', x))
    
    # Flatten list of all URLs
    all_urls = [url for sublist in df['urls'] for url in sublist]
    
    # Count frequencies
    counter = Counter(all_urls)
    counter_1 = counter + curr_counter
    
    # Get top URLs
    top_urls = Counter(dict(counter_1.most_common(top_common)))
    
    # Clean up
    del counter_1
    del counter
    
    return top_urls

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def clean_and_tokenize(text):
    # Lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    keywords = [word for word in tokens if word not in stop_words and word.isalpha()]
    
    return keywords

# Calculate Top Keywords
def top_keywords(df, curr_counter=Counter(), top_common=10):
    df = pretext_pipeline(df)
    df['keywords'] = df['text'].apply(lambda x: clean_and_tokenize(str(x)))
    
    # Flatten and count
    all_keywords = [word for sublist in df['keywords'] for word in sublist]
    counter = Counter(all_keywords)
    counter_1 = counter + curr_counter
    
    # Get top keywords
    top_kws = Counter(dict(counter_1.most_common(top_common)))
    
    del counter
    del counter_1
    
    return top_kws