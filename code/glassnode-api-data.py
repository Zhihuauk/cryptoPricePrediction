# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 03:41:01 2022

@author: song
"""

import numpy as np
import scipy as sp
import json as js
import requests as req
import pandas as pd
import datetime as dt
import os


#get one features data
def get_data_api(KEY,name,url):
    API_KEY = KEY
    cryptoName = name
    URL = url
    res = req.get(url,params = {'a':'ETH','i':'1w','api_key':API_KEY})
    return pd.read_json(res.text, convert_dates=['t'])
    

    
#df = get_data_api('24vuPNOyd3ciclDMDBqFCoQFykc','ETH','https://api.glassnode.com/v1/metrics/indicators/sopr')
  
#get multi-features data
def get_multi_features(KEY,namelst,urlst):
    dataFrame = []
    KEY = KEY
    lst1 = namelst
    lst2 = urlst
# =============================================================================
#     #os.makedirs('F:\ondeive\OneDrive - The University of Nottingham\研究生论文项目\data',exist_ok=True)
# =============================================================================
    for name in lst1:
        for url in lst2:
            labelName = url.split('/')[-1]
            
            df = get_data_api(KEY,name,url)
            df.set_index('t', inplace= True)
            df.rename(columns={'v':labelName},inplace =True)
            dataFrame.append(df)
            #df.to_csv(dataFile,sep = ',',encoding='utf-8')
            
    
  
    
  
KEY = '24vuPNOyd3ciclDMDBqFCoQFykc'    
namelst = ['BTC','ETH']
urlst = ['https://api.glassnode.com/v1/metrics/indicators/rhodl_ratio',
         'https://api.glassnode.com/v1/metrics/indicators/sopr',
         'https://api.glassnode.com/v1/metrics/indicators/hash_ribbon']


get_multi_features(KEY,namelst,urlst)
    

# =============================================================================
# "#apikey
# API_KEY = '24vuPNOyd3ciclDMDBqFCoQFykc'
# #request
# res = req.get('https://api.glassnode.com/v1/metrics/indicators/sopr',
#               params = {'a':'ETH',i:1w,'api_key':API_KEY})
# 
# df = pd.read_json(res.text, convert_dates={'t'})
# 
# #filter the aimed data
# df18_21 = df[df['t'].dt.year == 2020] 
# "
# =============================================================================



