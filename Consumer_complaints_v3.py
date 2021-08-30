#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion


# In[2]:


# from sklearn.feature_extraction.text.TfidfVectorizer import TfidfVectorizer


# In[3]:


train=pd.read_csv(r'/Users/gkhom4050/Documents/Training/Consumer_Complaints_train.csv')


# In[4]:


train.head(2)


# In[ ]:


# # drop ID, Sub-product, Company public response ( insufficient data)
# # Predicting company response to consumer 

# pipe 1 :   Impute ,Dummy - ZIP code(freq > 400),
# pipe 2: impute, dummy(cut off > 1000)
#     State( > 1000), Company > 1000, Product > 1000
#     Issue , Sub-issue, 
# Sub-product, Product

# pipe 3 : dummy & No imputation, no cutoff - Consumer disputed?, Timely response?, Submitted via, 
#     , Tags 
    
# pipe 4 :'Consumer consent provided?','Company response to consumer' : >4000  , 
        
    
# Numeric : Diff between date received and date sent to company

# Company public response( kind of response statements)

# # Text treatments pipe
# Consumer complaint narrative - remove XXXX


# In[ ]:


train.info()


# In[5]:


train.drop(['Complaint ID', 'Date sent to company', 'Date received'], axis = 1)


# In[ ]:


x='Tags'
train[x].value_counts()


# In[6]:


cat_vars_p1=['ZIP code']
cat_vars_p2=['State','Company','Product','Issue','Sub-issue','Sub-product']
cat_vars_p3=['Consumer disputed?','Timely response?','Submitted via','Tags']
cat_vars_p4=['Consumer consent provided?','Company response to consumer']
cat_vars_p5=['Consumer complaint narrative']


# In[13]:


import mpd


# In[14]:


from mpd import *


# In[ ]:


# pipe 1 :   Impute ,Dummy - ZIP code(freq > 400),
# pipe 2: impute, dummy(cut off > 1000)
#     State( > 1000), Company > 1000, Product > 1000
#     Issue , Sub-issue, 
# Sub-product, Product

# pipe 3 : dummy & No imputation, no cutoff - Consumer disputed?, Timely response?, Submitted via, 
#     , Tags 
    
# pipe 4 :'Consumer consent provided?','Company response to consumer' : >4000  , 
 


# In[16]:


p1= mpd.pdPipeline([
                ('var_select', VarSelector(cat_vars_p1)),
                ('missing_trt', DataFrameImputer()),
                ('Create_dummies', creat_dummies(400))
])

p2= mpd.pdPipeline([
                ('var_select', VarSelector(cat_vars_p2)),
                ('missing_trt', DataFrameImputer()),
                ('Create_dummies', creat_dummies(1000))
])

p3= mpd.pdPipeline([
                ('var_select', VarSelector(cat_vars_p3)),
                ('missing_trt', DataFrameImputer()),
                ('Create_dummies', creat_dummies(0))
])

p4= mpd.pdPipeline([
                ('var_select', VarSelector(cat_vars_p4)),
                ('missing_trt', DataFrameImputer()),
                ('Create_dummies', creat_dummies(4000))
])

p5= mpd.pdPipeline([
                ('var_select', VarSelector(cat_vars_p5)),
                ('string_clean', string_clean("XXXX","")),
                ('missing_trt', DataFrameImputer()),
                ('tfidf', tfidf())
#                 ('pdStdScaler',pdStdScaler())
])

# 5th pipe 


# In[17]:


data_pipe = FeatureUnion([
    ('p1',p1),
    ('p2',p2),
    ('p3',p3),
    ('p4',p4),
    ('p5',p5),
])


# In[18]:


data_pipe.fit(train)


# In[19]:


x_train = pd.DataFrame(data=data_pipe.transform(train),
                      columns = data_pipe.get_feature_names())


# In[ ]:




