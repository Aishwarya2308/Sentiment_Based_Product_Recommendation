#!/usr/bin/env python
# coding: utf-8

# ## Final Sentiment and Recommendation Model with  with the code to deploy the end-to-end project using Flask and Heroku

# In[1]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pickle as pkl
import numpy as np
from nltk.tokenize import word_tokenize
import xgboost


# ## import all pickle files
# #### xbg.pkl - sentiment analysis XGBoost model pickle file
# #### tfidf.pkl - tfidf vectorizer 
# #### transform.pkl - this pickle file after text cleaning
# #### user_recommendation.pkl - user based recommendation model

# In[3]:


bestmodel        = pkl.load(open('models/bestmodel.pkl','rb'))
tfidf      = pkl.load(open('models/tfidf.pkl','rb'))
transform  = pkl.load(open('dataset/transform.pkl','rb'))
user_recom = pkl.load(open('models/user_recommendation.pkl','rb'))


# In[4]:


def sentiment(recom_prod):
    df = transform[transform.name.isin(recom_prod)]
    features = tfidf.transform(df['text'])
    pred_data = bestmodel.predict(features)
    predictions = [round(value) for value in pred_data]
    df['predicted'] = predictions
    groupedDf = df.groupby(['name'])
    product_class = groupedDf['predicted'].agg(mean_class=np.mean)
    df = product_class.sort_values(by=['mean_class'], ascending=False)[:5]
    df['name'] = df.index
    data = df[['name']][:5].reset_index(drop=True)

    return data


# In[5]:


def recommendation(user_input):
    try:
        flag = True
        recom_data = user_recom.loc[user_input].sort_values(ascending=False)[0:20].index
    except:
        flag = False
        recom_data = "User  \"" + user_input + "\" not found. Please enter a valid user"
    return flag, recom_data
