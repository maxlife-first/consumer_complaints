#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

'''
class class_name(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        
        
        # add the necessary attributes 
        
    
    def fit(self,x,y=None):
        
        # learn from the data 
        # populate attributes 
        # learn featurenames 

        return self 
    
    def transform(self,x) :
        
        # make use of what you learnt in fit 
        # or othwerwise transform/modify dat

        return modified_data


        
    def get_feature_names(self) :
                
        return self.feature_names

'''
class convert_to_numeric(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        
        self.feature_names=[]
        
    
    def fit(self,x,y=None):

        self.feature_names=x.columns

        return self 
    
    def transform(self,x) :

        for col in x.columns:

            x[col]=pd.to_numeric(x[col],errors='coerce')

        return x
    
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

class custom_fico(BaseEstimator,TransformerMixin):

    def __init__(self):
        
        
        self.feature_names=['fico']
        
    
    def fit(self,x,y=None):

        return self 
    
    def transform(self,x) :
        
        k=x['FICO.Range'].str.split('-',expand=True).astype(int)
        fico=0.5*(k[0]+k[1])

        return pd.DataFrame({'fico':fico})

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

class custom_el(BaseEstimator,TransformerMixin):

    def __init__(self):
        
        
        self.feature_names=['EL']
        
    
    def fit(self,x,y=None):

        return self 
    
    def transform(self,x) :
        
        EL=x['Employment.Length'].str.replace('years','').str.replace('year','')
        EL=EL.replace({'10+ ':10,'< 1 ':0})

        return pd.DataFrame({'EL':EL})

    def get_feature_names(self) :
                
        return self.feature_names


class pdPipeline(Pipeline):

    def get_feature_names(self):

        last_step=self.steps[-1][-1]

        return last_step.get_feature_names()

class pdStdScaler(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.feature_names=[]
        self.std=StandardScaler()


    def fit(self,x,y=None):

        self.std.fit(x)
        self.feature_names=x.columns

        return self

    def transform(self,X):

        return(pd.DataFrame(data=self.std.transform(X),columns=self.feature_names))

    def get_feature_names(self):

        return self.feature_names

    
class tfidf(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        
        self.feature_names=[]
        self.model=TfidfVectorizer(analyzer='word',stop_words='english',max_df=0.8,min_df=0.01,max_features=200)
    
    def fit(self,x,y=None):

        self.feature_names=x.columns
        self.fitted=self.model.fit(x['col'])
        return self 
    
    def transform(self,x) :
        
        for col in x.columns:

            abc=seld.fitted.transform(x[col])
        
        return abc.toarray()

    def get_feature_names(self) :
                
        return self.feature_names
