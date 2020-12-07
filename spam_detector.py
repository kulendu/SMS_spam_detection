import nltk
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# nltk.download('stopwords')


messages = open('SMSSpamCollection')
messages = [ line.strip() for line in open('SMSSpamCollection', 'rU')] #so here 'rU' refers to 'r':reading the files and 'H': is for Universal, which is used for ignore the different conventions used for making newlines
#the strip function is used for removing the newline character at the end of each line.

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


# Vectorization --> Converting words to numbers.
from sklearn.feature_extraction.text import CountVectorizer

# 'bow' stands for Bag of words
bow_transformer = CountVectorizer(analyzer=clean_messages).fit(messages['messages'])
bow_messages = bow_transformer.transform(messages['messages'])

#sparsity -->
sparsity = (100.0 * bow_messages.nnz / (bow_messages.shape[0] * bow_messages.shape[1]))


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(bow_messages)
messages_tfidf = tfidf_transformer.transform(bow_messages)


# Model creation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['labels'])

#splitting the test and training data
train_msg, test_msg, train_labels, test_labels = train_test_split(messages['messages'], messages['labels'], test_size=0.3)

#creating an pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_messages)),  #bagOfWords analyzer
    ('tfidf', TfidfTransformer()),                      #tfidf-transformer
    ('classifier', MultinomialNB())                     #Classifier (the model/algo)
])
#fitting the data in the pipeline
pipeline.fit(train_msg, train_labels)

#making the predictions
pred = pipeline.predict(test_msg)


from sklearn.metrics import classification_report
print(classification_report(test_labels, pred))

