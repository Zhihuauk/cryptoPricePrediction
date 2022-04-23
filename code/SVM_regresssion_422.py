#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 22:30:58 2022

@author: song
"""
#-----------------------------------------------------------------------------#
#SVM 2 classes classification

# by using built-in package
from sklearn import svm, datasets 
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit,cross_val_score,cross_validate
from sklearn import metrics
import statistics as sta
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from time import time
from matplotlib import pyplot as plt

# df = pd.read_csv('C:/Users/song1/Desktop/MScProject/data/raw_col_norm.csv')
df = pd.read_csv('/Users/song/Desktop/MScProject/data/raw_col_norm.csv')
df_time = pd.DataFrame(df['time_real'])
df.set_index('time_real',inplace = True)

df_X = df.drop(columns = ['price_usd_close', 'price_usd_close.1','price_back_1_d', 'price_back_3_d', 'price_back_7_d',
       'price_back_15_d', 'price_back_30_d', 'SMA_3', 'SMA_7',
       'price_usd_close.2', 'price_usd_close.3', 'up_do_1d',
       'up_do_1d_2class'])
# 3 classification
# df_y = pd.DataFrame(df['up_do_1d_2class'])
# 2 classification 
df_y = pd.DataFrame(df['price_usd_close'])
X = df_X
y=df_y


#split dataset
split_percentage = 0.8
split = int(split_percentage*len(df))
# Train data set
X_train = X[:split]
y_train = y[:split]
# Test data set
X_test = X[split:]
y_test = y[split:]



from sklearn import metrics
from sklearn.svm import SVC,SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    #'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'kernel' : ['poly'],
    'degree':[1,2,3,4,5,6,7,8,9],
    'gamma':[0.01,0.05,0.1,0.25,0.5,0.7,1.0,2.0,5.0,10],
    'C':[0.1,0.5,1,5,10,50,100],
    'epsilon': [0.01,0.1,0.5,1.0,5.0]
}
                     
scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error','neg_root_mean_squared_error','r2']
                     
# Create a based model
rf = SVR()
# Instantiate the grid search model
grid_search = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, random_state=42,
                          cv = 5, n_jobs = 20, verbose = 2,scoring = scoring,n_iter = 10,refit='neg_mean_absolute_error')


# Fit the grid search to the data
grid_search.fit(X_train, y_train.values.ravel())
grid_search.best_params_

best_grid = grid_search.best_estimator_
best_para = grid_search.best_params_
best_accu = best_grid.score(X_test,y_test)
print('best_para is ',best_para)
print('test accuracy is ', best_accu)

plt.plot(X_test.index,y_test, c='b', label='Real values')
plt.plot(X_test.index,grid_search.predict(X_test), c='g', label='Prediction Values')
plt.legend()
plt.show()

result = pd.DataFrame(grid_search.cv_results_)
result.columns

result = pd.DataFrame(grid_search.cv_results_)
result = result[['mean_fit_time', 'params', 
       'mean_test_neg_mean_absolute_error',
       'std_test_neg_mean_absolute_error', 
                 'rank_test_neg_mean_absolute_error',
       'mean_test_neg_mean_squared_error', 'std_test_neg_mean_squared_error',
       'rank_test_neg_mean_squared_error', 
       'mean_test_neg_root_mean_squared_error',
       'std_test_neg_root_mean_squared_error',
       'rank_test_neg_root_mean_squared_error',]]
result.sort_values(by=['mean_test_neg_mean_squared_error'],ascending =False)