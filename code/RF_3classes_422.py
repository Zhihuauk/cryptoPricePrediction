#common package
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# by using built-in package
from sklearn import svm, datasets 
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
#for 
from sklearn.model_selection import ShuffleSplit,cross_val_score,cross_validate
from sklearn import metrics
import statistics as sta
from sklearn.metrics import recall_score
#for randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from time import time


df = pd.read_csv('/Users/song/Desktop/MScProject/data/raw_col_norm.csv')
df_time = pd.DataFrame(df['time_real'])
df.set_index('time_real',inplace = True)


df_X = df.drop(columns = ['price_usd_close', 'price_usd_close.1','price_back_1_d', 'price_back_3_d', 'price_back_7_d',
       'price_back_15_d', 'price_back_30_d', 'SMA_3', 'SMA_7',
       'price_usd_close.2', 'price_usd_close.3', 'up_do_1d',
       'up_do_1d_2class'])
# 3 classification
df_y = pd.DataFrame(df['up_do_1d'])
# 2 classification 
# df_y = pd.DataFrame(df['up_do_1d_2class'])
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10,20,30,40,50,60,70,80,90, 100, 120],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1,3, 4, 5],
    'min_samples_split': [2, 5, 12],
    'n_estimators': [100, 200, 500,700, 1000,1200,1500]
}
                     
from sklearn.metrics import make_scorer,precision_score,recall_score,balanced_accuracy_score,f1_score
# scoring = {'accuracy':'balanced_accuracy',
#            'precision_weight':make_scorer(precision_score(y_true, y_pred,average='weighted')),
#            'recall_weight':make_scorer(recall_score(y_true, y_pred,average='weighted')),
#            'f1':'f1_weighted'
#            }

scoring = {'accuracy':make_scorer(balanced_accuracy_score),
           'precision_weighted':make_scorer(precision_score,average='weighted'),
           'recall_weighted':make_scorer(recall_score,average='weighted'),
           'f1':make_scorer(f1_score,average='weighted')
           }
                     
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, random_state=42,
                          cv = 5, n_jobs = 16, verbose = 2,scoring = scoring,n_iter = 100,refit = 'accuracy')


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
result = result[['params', 
       'mean_test_accuracy',
       'std_test_accuracy', 'rank_test_accuracy',
       'mean_test_precision_weighted',
       'std_test_precision_weighted', 'rank_test_precision_weighted',
       'mean_test_recall_weighted',
       'std_test_recall_weighted', 'rank_test_recall_weighted',
       'mean_test_f1', 'std_test_f1', 'rank_test_f1']]
result.sort_values(by=['mean_test_accuracy'],ascending =False)