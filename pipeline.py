# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 14:53:59 2021

@author: skhom4875
"""

import warnings

import pandas as pd 
import numpy as np

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

from mypipes import *

p1=pdPipeline([
    ('var_select',VarSelector(['Date received' , 'Date sent to company' ])),
    ('convert_to_datetime',convert_to_datetime()),
    ('cyclic features',cyclic_features()),
    ('standard',pdStdScaler()),
    ('missing_trt',DataFrameImputer())
])

p2=pdPipeline([
    ('var_select',VarSelector(['Sub-Product', 'Sub-issue', 'Company public response','State', 'ZIP code',
                              'Tags', 'Consumer consent provided?'])),
    ('missing_trt',DataFrameImputer()),
    ('create_dummies',creat_dummies(20))
])

p3=pdPipeline([
    ('var_select',VarSelector([ 'Product', 'Issue', 'Submitted via', 'Company response to consumer', 'Timely response?','Consumer disputed?'])),
    ('create_dummies',creat_dummies(20))
])

p4=pdPipeline([
    ('var_select',VarSelector(['Consumer complaint narrative'])),
    ('vectorize',vectorize())
])

data_pipe=FeatureUnion([
    ('to_datetime',p1),
    ('dummy w/o missing v',p2),
    ('obj_to_dum',p3),
    ('vectorize',p4),
    ])

data_pipe.fit(df)