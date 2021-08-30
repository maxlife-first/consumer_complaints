# The Task 

We are going to be developing data processing pipelines for data set consumer_complaints_train.csv . 

you can download the data from here : https://www.dropbox.com/s/45uw7mbh7iv7ppw/Consumer_Complaints_train.csv

# Classes used 
[ add list of data processing classes that you will be making use of ]

* VarSelector 
* convert_to_datetime
* cyclic features
* tfidf
* DataFrameImputer
* string_clean
* creat_dummies

# Groups of Variable 
[Add different groups of variables here which will go through same data processing pipelines]
#   Column                        Non-Null Count   Dtype 
---  ------                        --------------   ----- 
 0   Date received                 478421 non-null  object
 1   Product                       478421 non-null  object
 2   Sub-product                   339948 non-null  object
 3   Issue                         478421 non-null  object
 4   Sub-issue                     185796 non-null  object
 5   Consumer complaint narrative  75094 non-null   object
 6   Company public response       90392 non-null   object
 7   Company                       478421 non-null  object
 8   State                         474582 non-null  object
 9   ZIP code                      474573 non-null  object
 10  Tags                          67206 non-null   object
 11  Consumer consent provided?    135487 non-null  object
 12  Submitted via                 478421 non-null  object
 13  Date sent to company          478421 non-null  object
 14  Company response to consumer  478421 non-null  object
 15  Timely response?              478421 non-null  object
 16  Consumer disputed?            478421 non-null  object
 17  Complaint ID                  478421 non-null  int64

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



# Notes : 

Dropping Complaint ID as it is unique and does not provide any insights from the data.
