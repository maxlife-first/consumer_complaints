#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import warnings
import mypipes
import pandas as pd 
import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.preprocessing import StandardScaler
#import CountVectorizer 
from sklearn.pipeline import Pipeline
from mypipes import *
warnings.filterwarnings('ignore')
#load data
mli_d1=pd.read_csv(r'D:\2. Learning\1. Trainings\Edvancer\Assignment1\Consumer_Complaints_train.csv')
variables={'V'+str(i+1):j for i,j in enumerate(mli_d1.columns)} # dictionaries
print(variables)
#selecting all variables other than Customer ID
features_selected=['Date received',
 'Product',
 'Sub-product',
 'Issue',
 'Sub-issue',
 'Consumer complaint narrative',
 'Company public response',
 'Company',
 'State',
 'ZIP code',
 'Tags',
 'Consumer consent provided?',
 'Submitted via',
 'Date sent to company',
 'Company response to consumer',
 'Timely response?',
 'Consumer disputed?']
df=VarSelector(features_selected) 
dataset=df.transform(mli_d1)
#converting 2 identified columns (v1, v4) to datetime
date_list=['Date received','Date sent to company']
dataset=convert_to_datetime().transform(dataset,date_list)
#imputing missing values for V2,V4,V7 v8,V9,v10, V11,v12,v13,v15,v16, v17,  V3, V5 - very low fill rate;
dataset=DataFrameImputer().fit(dataset).transform(dataset)
#creating dummies for V2,V4,V7 v8,V9,v10, V11,v12,v13,v15,v16, v17,  V3, V5 - very low fill rate;
dataset=create_dummies().fit(dataset).transform(dataset,features_impute_dummies)


# In[ ]:




