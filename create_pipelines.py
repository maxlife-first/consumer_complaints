#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# In[3]:


df=pd.read_csv("C:/Users/Manasi/Downloads/Consumer_Complaints_train.csv")


# In[55]:


df.shape


# In[6]:


df.info()


# In[8]:


{'V'+str(i+1):j for i,j in enumerate(df.columns)}


# In[ ]:


# V1 : convert to datetime and have cyclic features - p1
# V2,V8,V9,V13,V15,V16,V17 : create dummies -p2
# V3,V10,V11,V12 : impute missing,create dummies -p3
# V5,V7:  impute missing, use tfidfVectorizer from sklearn -p4
# V4,V6 : use tfidfVectorizer from sklearn -p5
# V14 : drop as date received and date sent to company are close
# V18 : drop 


# In[47]:


df["Consumer complaint narrative"].value_counts(dropna =False)


# ## making train and test dataset

# In[41]:


from sklearn.model_selection import train_test_split


# In[43]:


p_train, p_test= train_test_split(df, test_size=0.2,random_state=42)


# In[44]:


p_train.reset_index(drop=True,inplace=True)
p_test.reset_index(drop=True,inplace=True)


# ## Creating classes for pipelines

# In[54]:


class convert_to_datetime(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        
        self.feature_names=[]
        
    
    def fit(self,x,y=None):

        self.feature_names=x.columns

        return self 
    
    def transform(self,x) :

        for col in x.columns:

            x[col]=pd.to_datetime(x[col],errors='coerce')
            

        return x
    
    def get_feature_names(self) :
                
        return self.feature_names
    

class cyclic_features(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.feature_names=[]
        self.week_freq=7
        self.month_freq=12
        self.month_day_freq=31

    def fit(self,x,y=None):

        for col in x.columns:

            for kind in ['week','month','month_day']:

                self.feature_names.extend([col + '_'+kind+temp for temp in ['_sin','_cos']])

        return self 

    def transform(self,x):

        for col in x.columns:
            
            wdays=x[col].dt.dayofweek
            month=x[col].dt.month
            day=x[col].dt.day

            x[col+'_'+'week_sin']=np.sin(2*np.pi*wdays/self.week_freq)
            x[col+'_'+'week_cos']=np.cos(2*np.pi*wdays/self.week_freq)

            x[col+'_'+'month_sin']=np.sin(2*np.pi*month/self.month_freq)
            x[col+'_'+'month_cos']=np.cos(2*np.pi*month/self.month_freq)

            x[col+'_'+'month_day_sin']=np.sin(2*np.pi*day/self.month_day_freq)
            x[col+'_'+'month_day_cos']=np.cos(2*np.pi*day/self.month_day_freq)

            del x[col]

        return x

    def get_feature_names(self):

        self.feature_names



class VarSelector(BaseEstimator,TransformerMixin):
    
    def __init__(self,feature_names):
        
        
        self.feature_names=feature_names
        
    
    def fit(self,x,y=None):

        return self 
    
    def transform(self,x) :
        
        return x[self.feature_names]

    def get_feature_names(self) :
                
        return self.feature_names


class DataFrameImputer(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        
        
        self.feature_names=[]
        self.impute_dict={}
        
    
    def fit(self,x,y=None):
        
        self.feature_names=x.columns
        
        for col in x.columns:
            
            if x[col].dtype=='O':
                self.impute_dict[col]='missing'
            else :
                self.impute_dict[col]=x[col].median()
                
        return self
    
    def transform(self,x) :
        
        return x.fillna(self.impute_dict)
        
    def get_feature_names(self) :
                
        return self.feature_names

    
class string_clean(BaseEstimator,TransformerMixin):
    
    def __init__(self,replace_it,replace_with):
        
        
        self.feature_names=[]
        self.replace_it=replace_it
        self.replace_with=replace_with        
    
    def fit(self,x,y=None):

        self.feature_names=x.columns

        return self 
    
    def transform(self,x) :
        
        for col in x.columns:

            x[col]=x[col].str.replace(self.replace_it,self.replace_with)
        
        return x

    def get_feature_names(self) :
                
        return self.feature_names
    
    

class count_vectorizer(BaseEstimator,TransformerMixin):

    def __init__(self):
        
        
        self.feature_names=[]
        
    
    def fit(self,x,y=None):

        self.feature_names=x.columns
        
        return self
    
    def transform(self,x) :
        
        for col in x.columns:
            x[col]=x[col].TfidfVectorizer(analyzer='word',stop_words='english',max_df=0.8,min_df=0.01,max_features=200)
            x[col]=x[col].toarray()
            
        return  x
    
    def get_feature_names(self) :
                
        return self.feature_names
    
    

class standardise_num(BaseEstimator,TransformerMixin):
    
    
    def __init__(self,feature_names):
        
        
        self.feature_names=[]
        
    
    def fit(self,x,y=None):
        
        self.feature_names=x.columns
        
        scaler = StandardScaler()

        return self 
    
    def transform(self,x) :
                      
        scaled = scaler.fit_transform(x.columns)
        
        return scaled

    def get_feature_names(self) :
                
        return self.feature_names
                      
                      
class creat_dummies(BaseEstimator,TransformerMixin):
    
    def __init__(self,freq_cutoff=0):
        
        
        self.freq_cutoff=freq_cutoff
        self.var_cat_dict={}
        self.feature_names=[]
        
    
    def fit(self,x,y=None):

        for col in x.columns:

            k=x[col].value_counts()
            cats=k.index[k>=self.freq_cutoff][:-1]

            self.var_cat_dict[col]=list(cats)

        for col in self.var_cat_dict.keys():

            for cat in self.var_cat_dict[col]:
                self.feature_names.append(col+'_'+str(cat))

        return self 
    
    def transform(self,x) :
        
        dummy_data=x.copy()

        for col in self.var_cat_dict.keys():

            for cat in self.var_cat_dict[col]:

                name=col+'_'+str(cat)

                dummy_data[name]=(dummy_data[col]==cat).astype(int)

            del dummy_data[col]

        return dummy_data

    def get_feature_names(self) :
                
        return self.feature_names

    
class pdPipeline(Pipeline):

    def get_feature_names(self):

        last_step=self.steps[-1][-1]

        return last_step.get_feature_names()


# In[62]:


p1=pdPipeline([
    ('var_select',VarSelector(['Date received'])),
    ('convert_to_date', convert_to_datetime()),
    ('cyclic_feature',cyclic_features())
])

p2=pdPipeline([
    ('var_select',VarSelector(['Product','Submitted via','Company response to consumer','Timely response?',
                               'Company','State','Consumer disputed?'])),
    ('create_dummies',creat_dummies())
])

p3=pdPipeline([
    ('var_select',VarSelector(['Sub-product','ZIP code','Tags','Consumer consent provided?'])),
    ('missing_trt',DataFrameImputer()),
    ('create_dummies',creat_dummies())
])                               
                               
p4=pdPipeline([
    ('var_select',VarSelector(['Sub-issue','Issue','Company public response',])),
    ('missing_trt',DataFrameImputer()),
    ('text_convert',count_vectorizer())
])

p5=pdPipeline([
    ('var_select',VarSelector(['Consumer complaint narrative'])),
    ('string_clean',string_clean('XXXX','')),
    ('text_convert',count_vectorizer())   
])


# In[63]:


from sklearn.pipeline import FeatureUnion


# In[64]:


data_pipe = FeatureUnion([
    ('convert_date', p1),
    ('obj_to_dum',p2),
    ('miss_obj_to_dum',p3),
    ('miss_text_convert',p4),
    ('custom_text_convert',p5)
])


# In[66]:


data_pipe.fit(df)


# In[67]:


len(data_pipe.get_feature_names())


# In[69]:


data_pipe.transform(df).shape


# In[ ]:


x_train=pd.DataFrame(data=data_pipe.transform(train),
                    columns=data_pipe.get_feature_names())


# In[ ]:


x_test=pd.DataFrame(data=data_pipe.transform(test),
                    columns=data_pipe.get_feature_names())

