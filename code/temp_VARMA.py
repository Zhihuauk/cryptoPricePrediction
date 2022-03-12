# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:20:14 2022

@author: song
"""
%matplotlib inline

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random



data = pd.read_csv(r'G:\MScProject\data\norm_df.csv')
#ata = pd.read_csv(r'G:\MScProject\data\norm_df.csv').drop(['SMA_7'],axis=1)
#data = data[(data['Date'] > '2019-01-14') & (data['Date'] <= '2020-01-30')]

data['time_real']= pd.to_datetime(data['time_real'])


#data.set_index('time_real', inplace=True) 

data.info()
endog = data.iloc[200:500,-1:].set_index(data['time_real'][200:500])
endog = data.iloc[500:600,-1:].set_index(data['time_real'][500:600])
data_exog = data.iloc[200:500,-2:-1].set_index(data['time_real'][200:500])
data_exog2 = data.iloc[500:600,-2:-1].set_index(data['time_real'][500:600])



'''
dta = sm.datasets.webuse('lutkepohl2', 'https://www.stata-press.com/data/r12/')
dta.index = dta.qtr
dta.index.freq = dta.index.inferred_freq
endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]
'''


# fit model
model = VARMAX(data, exog=data_exog, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction

yhat = model_fit.forecast(exog=data_exog2)


#ax = res.impulse_responses()
#ax = res.impulse_responses(10, orthogonalized=True, impulse=[1, 0]).plot(figsize=(13,3))
#ax.set(xlabel='t', title='Responses to a shock to `dln_inv`')







