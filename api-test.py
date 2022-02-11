# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 04:30:59 2022

@author: song
"""

import numpy as np
import scipy as sp
import json as js
import requests as req
import pandas as pd
import datetime as dt
import os


KEY = '24vuPNOyd3ciclDMDBqFCoQFykc'    
namelst = ['BTC','ETH']
urlst = ['https://api.glassnode.com/v1/metrics/indicators/rhodl_ratio',
         'https://api.glassnode.com/v1/metrics/indicators/sopr',
         'https://api.glassnode.com/v1/metrics/indicators/hash_ribbon']


data = []
for url in urlst:
    label = url.split('/')[-1]
    
    res = req.get(url, params={'a':'ETH',
                                    'api_key':KEY})
    
df = pd.read_json(res.text, convert_dates=['t'])
df.set_index('t', inplace = True)
df.rename(columns ={'v':label}, inplace=True)
data.append(df)