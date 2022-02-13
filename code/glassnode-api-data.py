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
from  glassnode  import  GlassnodeClient


api_key = "24vuPNOyd3ciclDMDBqFCoQFykc"
client = GlassnodeClient(api_key)
since = 1589148000 # July 9 2016
until = 1768015200 # May 11 2020
resolution = "24h"
params = {"a": "BTC", "s": since, "u": until, "i": resolution}
#indictor list
indlist = [['market','price_usd_close'],
          # ['addresses','min_1_count'],
          # ['addresses','min_10_count'],
          # ['addresses','min_100_count'],
          # ['addresses','min_1k_count'],
          # ['addresses','min_10k_count'],
          # ['addresses','active_count']
          # ['addresses','new_non_zero_count'],
          #  ['addresses','receiving_count'],
           #cant use
           #['distribution','supply_contracts']
           ['distribution','balance_exchanges'],
           ['distribution','balance_exchanges_relative'],
           ['distribution','exchange_net_position_change'],
           ['distribution','supply_contracts'],
           ['transactions','transfers_volume_exchanges_net'],
           ['transactions','count'],
           ['supply','rcap_hodl_waves'],
           ['supply','active_3m_6m'],
           ['supply','active_6m_12m'],
           ['supply','active_1y_2y'],
           ['supply','active_2y_3y']
           # ['supply','active_3y_5y'],
           # ['supply','profit_relative'],
           # ['supply','profit_sum'],
           # ['supply','loss_sum'],
#            #do not have this data
#            #['supply','sth_lth_realized_value_ratio'],
#            ['distribution',''],
#            ['distribution',''],
#            ['distribution',''],
#            ['distribution',''],
#            ['distribution','supply_contracts'],
#            ['distribution','balance_exchanges'],
#            ['distribution','balance_exchanges_relative'],
#            ['distribution','exchange_net_position_change'],
#            ['distribution','supply_contracts'],
#            ['transactions','count'],
#            ['transactions','transfers_volume_exchanges_net'],
#            ['market','mvrv'],
#            ['market','mvrv_z_score'],
#            ['market','marketcap_realized_usd'],
#            ['indicators','pi_cycle_top'],
#            ['indicators','net_unrealized_profit_loss'],
#            ['indicators','seller_exhaustion_constant'],
#            ['indicators','realized_profits_to_value_ratio'],
#            ['indicators','realized_profit'],
#            ['indicators','realized_loss'],
#            ['indicators','ssr'],
#            ['indicators','ssr_oscillator'],
#            ['lightning','network_capacity_sum'],
#            ['protocols','uniswap_volume_sum']
#            #['',''],
            
          ]

# indlist = [['indicators','realized_profit'],
#            ['supply','rcap_hodl_waves'],
#            ['lightning','network_capacity_sum'],
#            ['indicators','realized_profits_to_value_ratio'],
#            ['indicators','realized_loss'],
#            ['indicators','ssr'],
#            ['indicators','ssr_oscillator'],
#            ['supply','sth_lth_realized_value_ratio']
#           ]

#main function get data from api and transfor to dataframe
data=[]
for ind in indlist:
    label = ind[1]
    db = client.get(ind[0],ind[1],params)
    #convert from list of dictionaries to data frame
    df = pd.DataFrame.from_dict(db,orient='columns')
    #setup columns name
    df.columns=['time',label]
    #fixed index
    df.set_index('time', inplace = True)
    data.append(df)
    data_df = pd.concat(data,axis = 1)
