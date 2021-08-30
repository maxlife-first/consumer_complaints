#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer



class VarSelector(BaseEstimator,TransformerMixin):
    
    def __init__(self,feature_names):
        
        
        self.feature_names=feature_names
        
    
    def fit(self,x,y=None):

        return self 
    
    def transform(self,x) :
        
        return x[self.feature_names]

    def get_feature_names(self) :
                
        return self.feature_names
    
    


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

    
class tfidf(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        
        self.feature_names=[]
        
        vectorizer = TfidfVectorizer(analyzer='word',stop_words='english',max_df=0.8,min_df=0.01,max_features=200)
    
    def fit(self,x,y=None):

        self.feature_names=x.columns
        
        return self 
    
    def transform(self,x) :
              
        for col in x.columns:
            corpus = col
            df = vectorizer.fit_transform(corpus)
            b = vectorizer.get_feature_names()
            df = df.toarray()
            df = pd.DataFrame(df)
            df.columns = b
            del x[col]
            

        return df

    def get_feature_names(self) :
                
        return self.feature_names
    

class pdPipeline(Pipeline):

    def get_feature_names(self):

        last_step=self.steps[-1][-1]

        return last_step.get_feature_names()








