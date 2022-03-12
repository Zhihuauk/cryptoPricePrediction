# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:42:13 2022

@author: song
"""
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

#dataset
#df = pd.read_csv(r'G:\MScProject\data\raw_col_norm.csv')
df.dropna(inplace = True)
x = df.iloc[:,2:35].drop(columns = ['rcap_hodl_waves','pi_cycle_top','ssr'])
y = df.iloc[:,-1:]
# y = df['up_do_1d']

# x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,train_size = 0.80,test_size=0.2,random_state = 101)

# #kernal
# rbf = svm.SVC(kernel = 'rbf',gamma=0.5,C =0.1).fit(x_train,y_train)
# poly = svm.SVC(kernel='poly',degree=3,C=1).fit(x_train,y_train)

# rbf_p = rbf.predict(x_test)
# poly_p = poly.predict(x_test)

# poly_acc = accuracy_score(y_test,poly_p)
# poly_f1 = f1_score(y_test,poly_p, average='weighted')
# print('acccuracy(ploy):','%.2f' %(poly_acc*100))
# print('F1 (RBF kernal):: ','%.2f' %(poly_f1*100))

# rbf_acc = accuracy_score(y_test,rbf_p)
# rbf_f1 = f1_score(y_test,rbf_p, average='weighted')
# print('acccuracy(ploy):','%.2f' %(rbf_acc*100))
# print('F1 (RBF kernal):: ','%.2f' %(rbf_f1*100))




def SVM_fun(x_train,x_test,y_train,y_test):
    
    
    poly = svm.SVC(kernel='poly',degree=3,C=1).fit(x_train,y_train)
    poly_p = poly.predict(x_test)
    poly_acc = accuracy_score(y_test,poly_p)
    poly_f1 = f1_score(y_test,poly_p, average='weighted')
    

    
    rbf = svm.SVC(kernel = 'rbf',gamma=0.5,C =0.1).fit(x_train,y_train)
    rbf_p = rbf.predict(x_test)
    rbf_acc = accuracy_score(y_test,rbf_p)
    rbf_f1 = f1_score(y_test,rbf_p, average='weighted')
   
    return [[[poly_p],[rbf_p]]]
    # return [[poly_acc,poly_f1,rbf_acc,rbf_f1]]


    
    
#cross validation
l = len(df)
folds= 10
fold_len = l//folds
#list record the result
#Metrics_df = pd.DataFrame()
result = []


for i in range(1,10):
    x_train = x.iloc[0:i*fold_len,:]
    x_test = x.iloc[i*fold_len:(i+1)*fold_len,:]
    y_train = y.iloc[0:i*fold_len,:]
    y_test = y.iloc[i*fold_len:(i+1)*fold_len,:]
    res = SVM_fun(x_train,x_test,y_train,y_test)
    
    result += res
   
    

result
 
    





