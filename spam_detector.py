import nltk
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# nltk.download('stopwords')


messages = open('SMSSpamCollection')
messages = [ line.strip() for line in open('SMSSpamCollection', 'rU')] #so here 'rU' refers to 'r':reading the files and 'H': is for Universal, which is used for ignore the different conventions used for making newlines
#the strip function is used for removing the newline character at the end of each line.

# for msg_no, msg in enumerate(messages[:10]):  #giving the message number to the first 10 messages and enmerating through it
#   print(msg_no,msg)
#   print('\n')

#converting the normal tsv(tab-separated-value) to csv file
messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['labels','messages'])

#creating a new columns for the length of the msgs
messages['length'] = messages['messages'].apply(len)
# print(messages.head())



import string
from nltk.corpus import stopwords #importing the stopwords.Stopwords are words that doesnt tell us any distinguishing meaning.

'''
Tokenization of the messages, what is tokens?
--> Tokenization is the process of:
    1. Removing punctuation
    2. Removing stopwords(words like: if ,once, had...)
    3. Returning list of clean text words
'''

def clean_messages(mess):   #function for cleaning the messages.
    no_punc = [char for char in mess if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

messages['messages'].apply(clean_messages)
print(messages.head())


# Vectorization --> Converting words to numbers.
from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=clean_messages).fit(messages['messages'])
