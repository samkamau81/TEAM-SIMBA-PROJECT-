#!/usr/bin/env python
# coding: utf-8

# # Data Collection 

# In[1]:


import tweepy
import pandas as pd
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from wordcloud import WordCloud
import nltk 
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
from nltk.corpus import stopwords
nltk.download("stopwords")


# In[ ]:


CONSUMER_KEY = "xxxxxxxxxxxxxxxx"
CONSUMER_SECRET = "xxxxxxxxxxxxxxxxxxxxxx" 
ACCESS_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ACCESS_TOKEN_SECRET = "xxxxxxxxxxxxxxxxxxxxxxxxxx"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)


# In[ ]:



api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Enter your search words in accordance with the basic filtering rules
search_words = "(coronavirus OR covid OR pandemic OR covid19 OR lockdown) AND ( loneliness OR lonely OR depressed OR suicide OR sad) "

# We also want to exclude retweets and replies as this may sway results
my_search = search_words + " -filter:retweets" + " -filter:replies"  


# In[ ]:


# The Twitter data is stored in a Tweet object which we've called tweets
tweets = api.search(q=my_search,lang="en",tweet_mode="extended",count=100)
# Iterate and print tweets
i = 1
for tweet in tweets[0:20]:
    print(str(i) + ') ' + tweet.full_text + '\n')
    i = i + 1 


# In[ ]:


# Our new method of collecting the tweets
tweets = tweepy.Cursor(api.search,q=my_search,lang="en",tweet_mode='extended').items(1000)


# In[ ]:


# Extract the info we need from the tweets object
tweet_info = [[tweet.id_str,tweet.created_at,tweet.user.location,tweet.full_text] for tweet in tweets]


# In[ ]:


# Put our data into a dataframe 
df = pd.DataFrame(data=tweet_info, columns=['tweet_id_str','date_time','location','tweet_text'])

# Have a quick look at the dataframe
df


# In[ ]:


#for i,tweet in enumerate(df['tweet_text'].head(20)):
   # print(i+1, tweet, '\n')


# # Data Cleaning and Processing 

# In[ ]:


def clean_text(text):
    
    """
    A function to clean the tweet text
    """
    #Remove hyper links
    text = re.sub(r'https?:\/\/\S+', ' ', text)
    
    #Remove @mentions
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)
    
    #Remove anything that isn't a letter, number, or one of the punctuation marks listed
    text = re.sub(r"[^A-Za-z0-9#'?!,.]+", ' ', text)   
    
    return text


# In[ ]:


# Apply the clean_text function to the 'tweet_text' column
df['tweet_text']=df['tweet_text'].apply(clean_text)

#for i,tweet in enumerate(df['tweet_text'].head(20)):
    #print(i+1, tweet, '\n')


# In[ ]:


df['tweet_text']=df['tweet_text'].str.lower()


# In[ ]:


# Get the list of NLTK stop words

stopwords = stopwords.words("english")


# In[ ]:


# Define our own list of stopwords
my_stopwords = ['coronavirus','covid','pandemic','covid19','lockdown','amp','via']

# Extend the nltk stopwords list
stopwords.extend(my_stopwords)


# In[ ]:


def remove_stopwords(text):
    
    """
    A function to remove stop words
    """
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    
    return filtered_text


# In[ ]:


# Apply the stopword removal function to the text of all tweets
df['tweet_text']=df['tweet_text'].apply(remove_stopwords)


# In[ ]:


# Plot a word cloud

all_words = ' '.join( [data for data in df['tweet_text']])
word_cloud = WordCloud(width=300, height=200, random_state=21, max_font_size = 300,
                       stopwords=stopwords).generate(all_words)

plt.figure(figsize = (20,10))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
#plt.show()


# In[ ]:


df.to_csv('Simba_tweets.csv')


# # Using ML

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from simpletransformers.classification import ClassificationModel

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch
from sklearn.model_selection import KFold

import logging
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report 


# In[ ]:


df2 = pd.read_csv("labelled_data_preprocessed.csv")
# Display some useful information about the data in the dataframe
df2.info()


# In[ ]:


# Drop all rows which contain a NaN, or null value in either column
df2= df2.dropna()


# In[ ]:


df2['label'].value_counts()


# In[ ]:


# make it binary classification challenge
binary_df = df2[(df2['label']==0) | (df2['label']==2) ] 


# In[ ]:


# Change the label for positive sentiment from 2 to 1
binary_df['label'] = binary_df['label'].replace(2, 1) 


# In[ ]:


binary_df['label'].value_counts()


# In[ ]:


binary_df = df2[(df2.label==0) | (df2.label==2) ] # make it binary classification
binary_df.label.replace(2,1, inplace=True) # make it binary classification

pos_samples =binary_df[binary_df['label']==1]
neg_samples = binary_df[binary_df['label']==0].sample(len(pos_samples), random_state=42)

bal_binary_df = pd.concat([pos_samples, neg_samples])

binary_df['label'].value_counts()
bal_binary_df['label'].value_counts()


# In[ ]:


binary_df.value_counts()


# In[ ]:


train_df, val_df = train_test_split(binary_df, test_size=0.2,  random_state=42)


# In[ ]:


train_df['label'].value_counts()
val_df['label'].value_counts()


# # BERT

# In[ ]:


get_ipython().run_cell_magic('time', '', "# ___Cell no. 30___\n\n# Build the model\n\nbert_model = ClassificationModel('bert',\n                            'bert-base-cased',\n                            num_labels=2,\n                            use_cuda=False,\n                            args={'overwrite_output_dir': True})")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# ___Cell no. 31___\n\n# Train the model \nbert_model.train_model(train_df=train_df, eval_df=val_df)')


# In[ ]:


# wrapper functions

def multi_F1(y_true, y_pred, average='macro'):
    return sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred, average=average)

def multi_classification_report(y_true, y_pred):
    return sklearn.metrics.classification_report(y_true=y_true, y_pred=y_pred)


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# ___Cell no. 33___\n\n# Calculated and print out the f1 score\n\nresult, model_outputs, wrong_predictions = bert_model.eval_model(val_df, f1=multi_F1);\nprint('f1 score = ',result['f1'])")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# ___Cell no. 34___\n\n# Calculated and print out the results in the classification report\n\nresult, model_outputs, wrong_predictions = bert_model.eval_model(val_df, report=multi_classification_report);\nprint('Classification Report: ', result['report'])")


# In[ ]:


tweets = df['tweet_text']


# In[ ]:


tweets.shape


# In[ ]:


Result = bert_model.predict(tweets)
print(Result[0])


# In[ ]:


df['Bert_Sentiment'] = Result[0]


# In[ ]:


def get_sentiment_label(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'    


# In[ ]:


df


# In[ ]:


df['Bert_sntment']=df['Bert_Sentiment'].apply(get_sentiment_label)


# In[ ]:


df


# In[ ]:


del df['tweet_id_str']


# In[ ]:


df['Bert_sntment'].value_counts().plot(kind='bar',ylabel='People',title='Mental Health Sentiment Analysis around the World',figsize=(8,8))


# # East African Region

# In[ ]:


loc = "-0.0236,37.9062,1000km"

# Search dates
date_since = "2021-07-22"
date_until = "2021-07-29"
search_words_2="(coronavirus OR covid OR pandemic OR covid19 OR lockdown OR corona OR mask ) AND ( depression OR loneliness OR lonely OR depressed OR suicide OR sad OR dryspell OR suicidal OR unhappy) "
# We also want to exclude retweets and replies as this may sway results
my_search = search_words_2 + " -filter:retweets" + " -filter:replies"  
#for i,tweet in enumerate(df['tweet_text'].head(20)):
   # print(i+1, tweet, '\n')
    
# Use the tweepy Cursor method to access tweets from a specified region and between certain dates
tweets = tweepy.Cursor(api.search,
                       q=my_search,
                       lang="en",
                       tweet_mode='extended',
                       geocode=loc,
                       since=date_since,
                       until=date_until).items(1000)

tweet_info = [[tweet.id_str,tweet.created_at,tweet.user.location,tweet.full_text] for tweet in tweets]

# Put our data into a dataframe 
df_new = pd.DataFrame(data=tweet_info, columns=['tweet_id_str','date_time','location','tweet_text'])
df_new


# In[ ]:


df_new['tweet_text']=df_new['tweet_text'].apply(clean_text)
df_new['tweet_text']=df_new['tweet_text'].str.lower()
df_new['tweet_text']=df_new['tweet_text'].apply(remove_stopwords)
all_words = ' '.join( [data for data in df_new['tweet_text']])
word_cloud = WordCloud(width=300, height=200, random_state=21, max_font_size = 300,
                       stopwords=stopwords).generate(all_words)

plt.figure(figsize = (20,10))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


tweets = df_new['tweet_text']
Result = bert_model.predict(tweets)
print(Result[0])


# In[ ]:


df_new['Bert_Sentiment'] = Result[0]


# In[ ]:


df_new


# In[ ]:


df_new['Bert_sntment'] = df_new['Bert_Sentiment'].apply(get_sentiment_label)


# In[ ]:


df_new


# In[ ]:


df_new['Bert_sntment']
df_new['Bert_sntment'].value_counts().plot(kind='bar',ylabel='Number of tweets',title='Bar chart of East Africa Region');


# In[ ]:


fig=plt.figure()
ax0=fig.add_subplot(1,2,1)
ax1=fig.add_subplot(1,2,2)

df['Bert_Sentiment'].value_counts().plot(kind='barh',color='blue',figsize=(10,10))
ax0.set_title('Bar chart of the World')
ax0.set_ylabel('Number of tweets')
plt.savefig('simba1', dpi=360)


df_new['Bert_Sentiment'].value_counts().plot(kind='barh',color='blue',figsize=(10,10))
ax1.set_title('Bar chart of East Africa Region')
ax1.set_ylabel('Number of tweets')


# In[ ]:


df['location'].dropna(inplace=True)

