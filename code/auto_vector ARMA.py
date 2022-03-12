# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 07:43:05 2022

@author: song
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn import metrics
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")



#-----------------------------------------------------------------------------#
# Import libraries
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
# Generate a sample dataset with correlated variables

# fit model
model = VARMAX(data, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.forecast()
print(yhat)
#-----------------------------------------------------------------------------#
data = pd.read_csv(r'G:\MScProject\data\norm_df.csv')
#ata = pd.read_csv(r'G:\MScProject\data\norm_df.csv').drop(['SMA_7'],axis=1)
#data = data[(data['Date'] > '2019-01-14') & (data['Date'] <= '2020-01-30')]

data['time_real']= pd.to_datetime(data['time_real'])


data.set_index('time_real', inplace=True) 

data.info()


#data['time_real']

data.isnull().sum()

#-----------------------------------------------------------------------------#
def timeseries_evaluation_metrics_func(y_true, y_pred):
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print('Evaluation metric results:-')
    print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
    print(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
    print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
    print(f'MAPE is : {mean_absolute_percentage_error(y_true, y_pred)}')
    print(f'R2 is : {metrics.r2_score(y_true, y_pred)}',end='\n\n')
    
    
    
#-----------------------------------------------------------------------------#    
def Augmented_Dickey_Fuller_Test_func(series , column_name):
    print (f'Results of Dickey-Fuller Test for column: {column_name}')
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','No Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    if dftest[1] <= 0.05:
        print("Conclusion:")
        print("Reject the null hypothesis")
        print("Data is stationary")
    else:
        print("Conclusion:")
        print("Fail to reject the null hypothesis")
        print("series is non-stationary")

#-----------------------------------------------------------------------------#       
for name, column in data[data.columns].iteritems():
    Augmented_Dickey_Fuller_Test_func(data[name],name)
    print('\n')
    
#-----------------------------------------------------------------------------#   
X = data[data.columns]
l = len(data)
split_point = int(l*0.8)
train, test = X[0:split_point], X[split_point:]

#-----------------------------------------------------------------------------#

train_diff = train.diff()
train_diff.dropna(inplace = True)

#-----------------------------------------------------------------------------#
for name, column in train_diff[data.columns].iteritems():
    Augmented_Dickey_Fuller_Test_func(train_diff[name],name)
    print('\n')
    
    
#-----------------------------------------------------------------------------#    
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df): 
    res = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = res.lr1
    cvts = res.cvt[:, d[str(1-0.05)]]
    def adjust(val, length= 6): 
        return str(val).ljust(length)
    print('Column Name   >  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), '> ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
        
        
        
# SAM_7 is not used here
cointegration_test(train_diff[['active_count', 'new_non_zero_count', 'receiving_count',
       'balance_exchanges', 'balance_exchanges_relative',
       'exchange_net_position_change', 'active_3m_6m', 'active_6m_12m',
       'active_1y_2y', 'active_2y_3y', 'active_3y_5y', 'profit_relative',
       'profit_sum', 'loss_sum', 'count', 'transfers_volume_exchanges_net',
       'mvrv', 'mvrv_z_score', 'marketcap_realized_usd',
       'net_unrealized_profit_loss', 'seller_exhaustion_constant',
       'realized_profits_to_value_ratio', 'realized_profit', 'realized_loss',
       'ssr_oscillator', 'network_capacity_sum', '1_10_address',
       '10-100-address', '100-1k-address', '1k-10k-address', '10k+address','SMA_3','price_usd_close'
       ]])
#-----------------------------------------------------------------------------#

from pmdarima import auto_arima

pq = []
for name, column in train_diff[[ 'Open', 'High', 'Low', 'Close'  ]].iteritems():
    print(f'Searching order of p and q for : {name}')
    stepwise_model = auto_arima(train_diff[name],start_p=1, start_q=1,max_p=7, max_q=7, seasonal=False,
        trace=True,error_action='ignore',suppress_warnings=True, stepwise=True,maxiter=1000)
    parameter = stepwise_model.get_params().get('order')
    print(f'optimal order for:{name} is: {parameter} \n\n')
    pq.append(stepwise_model.get_params().get('order'))
    
#-----------------------------------------------------------------------------#    
    
def inverse_diff(actual_df, pred_df):
    df_res = pred_df.copy()
    columns = actual_df.columns
    for col in columns: 
        df_res[str(col)+'_1st_inv_diff'] = actual_df[col].iloc[-1] + df_res[str(col)].cumsum()
    return df_res

pq
#-----------------------------------------------------------------------------#

df_results_moni = pd.DataFrame(columns=['p', 'q','RMSE Open','RMSE High','RMSE Low','RMSE Close'])
print('Grid Search Started')
start = timer()
for i in pq:
    if i[0]== 0 and i[2] ==0:
        pass
    else:
        print(f' Running for {i}')
        model = VARMAX(train_diff[[ 'Open', 'High', 'Low', 'Close'   ]], order=(i[0],i[2])).fit( disp=False)
        result = model.forecast(steps = 30)
        inv_res = inverse_diff(df[[ 'Open', 'High', 'Low', 'Close'   ]] , result)
        Opensrmse = np.sqrt(metrics.mean_squared_error(test['Open'], inv_res.Open_1st_inv_diff))
        Highrmse = np.sqrt(metrics.mean_squared_error(test['High'], inv_res.High_1st_inv_diff))
        Lowrmse = np.sqrt(metrics.mean_squared_error(test['Low'], inv_res.Low_1st_inv_diff))
        Closermse = np.sqrt(metrics.mean_squared_error(test['Close'], inv_res.Close_1st_inv_diff))
        df_results_moni = df_results_moni.append({'p': i[0], 'q': i[2], 'RMSE Open':Opensrmse,'RMSE High':Highrmse,'RMSE Low':Lowrmse,'RMSE Close':Closermse }, ignore_index=True)
end = timer()
print(f' Total time taken to complete grid search in seconds: {(end - start)}')
#-----------------------------------------------------------------------------#



df_results_moni.sort_values(by = ['RMSE Open','RMSE High','RMSE Low','RMSE Close'] )

# from above example we can see that p=0 and q=2 gives least RMSE
model = VARMAX(train_diff[[ 'Open', 'High', 'Low', 'Close' ]], order=(0,2)).fit( disp=False)
result = model.forecast(steps = 30)


res = inverse_diff(df[['Open', 'High', 'Low', 'Close' ]],result)
res

for i in ['Open', 'High', 'Low', 'Close' ]:
    print(f'Evaluation metric for {i}')
    timeseries_evaluation_metrics_func(test[str(i)] , res[str(i)+'_1st_inv_diff'])
    
    
    
import matplotlib.pyplot as plt
%matplotlib inline
for i in ['Open', 'High', 'Low', 'Close' ]:
    
    plt.rcParams["figure.figsize"] = [10,7]
    plt.plot( train[str(i)], label='Train '+str(i))
    plt.plot(test[str(i)], label='Test '+str(i))
    plt.plot(res[str(i)+'_1st_inv_diff'], label='Predicted '+str(i))
    plt.legend(loc='best')
    plt.show()