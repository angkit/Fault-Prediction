import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.svm import SVC, LinearSVC
import csv
# Training data
# Data preparation with pandas


df1= pd.read_csv('train.csv')
df2= pd.read_csv('log_feature.csv')
df3=pd.read_csv('resource_type.csv')
df4=pd.read_csv('fault_type.csv')
df5=pd.read_csv('event_type.csv')



result = pd.merge(df1,df2, on='id')
result = pd.merge(result,df3, on='id')
result = pd.merge(result,df4, on='id')
result = pd.merge(result,df5, on='id')
#print(result.head())
result=result.drop_duplicates(keep='first')
#print(result.head())


X_train = result[['location','log_feature','resourve_type','type_of_faults','event_type']]
#print(x_train.head())
Y_train = result[['fault_severity']]
#print(y_train.head())



#print(X_train.shape)
#print(Y_train.shape)
# Testing data

d1= pd.read_csv('test.csv')
d2= pd.read_csv('log_feature.csv')
d3=pd.read_csv('resource_type.csv')
d4=pd.read_csv('fault_type.csv')
d5=pd.read_csv('event_type.csv')



result1 = pd.merge(d1,d2, on='id')
result1 = pd.merge(result1,d3, on='id')
result1 = pd.merge(result1,d4, on='id')
result1 = pd.merge(result1,d5, on='id')
result1=result1.drop_duplicates(keep='first')

#print(result1.head())
X_test = result1[['location','log_feature','resourve_type','type_of_faults','event_type']]
#print(X_test.shape)

#cat_train=X_train.head(20) 
#cat_test=X_test.head(10)

#Y_train=Y_train.head(20)

#print(Y_train.shape)
#print(Y_train)


x_train = X_train.to_dict( orient = 'records' )
x_test = X_test.to_dict( orient = 'records' )

# vectorize

vectorizer = DV( sparse = False )
vec_x_train = vectorizer.fit_transform( x_train )
vec_x_test = vectorizer.transform( x_test )




log=LogisticRegression(penalty='l2',C=1,class_weight='balanced')
log.fit(vec_x_train,Y_train.values.ravel())



p = log.predict(vec_x_test)


with open("output.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(p)

#LinearSVC_classifier.fit(vec_x_cat_train,Y_train.values.ravel())

#p=LinearSVC_classifier.predict_proba(vec_x_cat_test)
#print(p)