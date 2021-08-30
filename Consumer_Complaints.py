#!/usr/bin/env python
# coding: utf-8

# In[89]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd 
import numpy as np

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from consumer_pipes import *


# In[5]:


train=pd.read_csv('Consumer_Complaints_train.csv')


# In[6]:


train.info()


# In[7]:


train.sample(10)


# In[ ]:


# A=Varselector
# B=datetime
# C=tf-idfvectorizer
# D=missing
# E=create dummies
# F=cyclic
# H=string_clean

# 0,13 = A,B,F,D #date_vars
# 1,2,7,8,9,10,11,12,14,15,16 = A,D,E #cat_vars
# 3,4,6 = A,C #text_vars
# 5 = A,H,C #Consumer complaint narrative


# In[79]:


date_vars=['Date received','Date sent to company']


# In[80]:


text_vars=['Issue','Sub-issue','Company public response']


# In[81]:


cat_vars=train.select_dtypes(include=['object']).columns

cat_vars=[_ for _ in cat_vars if _ not in text_vars]
cat_vars.remove('Date received')
cat_vars.remove('Date sent to company')
cat_vars.remove('Consumer complaint narrative')


# In[82]:


complaint=['Consumer complaint narrative']


# In[83]:


len(cat_vars)+len(date_vars)+len(text_vars)+len(complaint)


# In[28]:


p_train,p_test=train_test_split(train,test_size=0.2,random_state=42)


# In[29]:


p_train.reset_index(drop=True,inplace=True)
p_test.reset_index(drop=True,inplace=True)


# In[90]:


p1=pdPipeline([
    ('var_select',VarSelector(date_vars)),
    ('convert_to_date',convert_to_datetime()),
    ('cyclic_features',cyclic_features()),
    ('missing_trt',DataFrameImputer())
])

p2=pdPipeline([
    ('var_select',VarSelector(cat_vars)),
    ('missing_trt',DataFrameImputer()),
    ('create_dummies',creat_dummies(200))
])

p3=pdPipeline([
    ('var_select',VarSelector(text_vars)),
    ('tfidf',tfidf())
])

p4=pdPipeline([
    ('var_select',VarSelector(complaint)),
    ('string_clean',string_clean(replace_it='XXXX',replace_with='')),
    ('tfidf',tfidf())
])


data_pipe=FeatureUnion([
    ('p1',p1),
    ('p2',p2),
    ('p3',p3),
    ('p4',p4)    
])


# In[91]:


data_pipe.fit(p_train)


# In[ ]:




