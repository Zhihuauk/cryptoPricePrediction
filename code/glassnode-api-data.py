# import package
from  glassnode  import  GlassnodeClient
import pandas as pd
import until
import seaborn as sns
import matplotlib.pyplot as plt




#part1
#initialise paramters
#-----------------------------------------------------------------------------
#-------------------------------------------------------------------------------

api_key = "24vuPNOyd3ciclDMDBqFCoQFykc"
client = GlassnodeClient(api_key)
since = 1589148000 # July 9 2016
until = 1768015200 # May 11 2020
resolution = "24h"
params = {"a": "BTC", "s": since, "u": until, "i": resolution}
#indicators list
indlist = [['market','price_usd_close'],
          ['addresses','min_1_count'],
          ['addresses','min_10_count'],
          ['addresses','min_100_count'],
          ['addresses','min_1k_count'],
          ['addresses','min_10k_count'],
          ['addresses','active_count'],
          ['addresses','new_non_zero_count'],
           ['addresses','receiving_count'],
           #cant use
           ['distribution','supply_contracts'],
           ['distribution','balance_exchanges'],
           ['distribution','balance_exchanges_relative'],
           ['distribution','exchange_net_position_change'],
           #cant use
           ['supply','rcap_hodl_waves'],
           ['supply','active_3m_6m'],
           ['supply','active_6m_12m'],
           ['supply','active_1y_2y'],
           ['supply','active_2y_3y'],
           ['supply','active_3y_5y'],
           ['supply','profit_relative'],
           ['supply','profit_sum'],
           ['supply','loss_sum'],
           #do not have this data
           ['supply','sth_lth_realized_value_ratio'],
           #cant use
           ['transactions','count'],
           ['transactions','transfers_volume_exchanges_net'],
           ['market','mvrv'],
           ['market','mvrv_z_score'],
           ['market','marketcap_realized_usd'],
           ['indicators','pi_cycle_top'],
           ['indicators','net_unrealized_profit_loss'],
           ['indicators','seller_exhaustion_constant'],
           ['indicators','realized_profits_to_value_ratio'],
           ['indicators','realized_profit'],
           ['indicators','realized_loss'],
           ['indicators','ssr'],
           ['indicators','ssr_oscillator'],
           ['lightning','network_capacity_sum'],
           #cant use this data
           ['protocols','uniswap_volume_sum']
           #['',''],          
          ]
#part2
#get data from glossnode api
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def data_from_api(indlist):
    data=[]
    for i in range(len(indlist)):
        try:
            label = indlist[i][1]
            db = client.get(indlist[i][0],indlist[i][1],params)
            #convert from list of dictionaries to data frame
            df = pd.DataFrame.from_dict(db,orient='columns')
            #setup columns name
            df.columns=['time',label]
            #fixed index
            df.set_index('time', inplace = True)
            data.append(df)
            data_df = pd.concat(data,axis = 1)
        except:
            print('data can be used',label)
    data_df['time'] = data_df.index
    data_df['time'] = pd.to_datetime(data_df['time'],unit = 's')
    #set time as the first columns
    data_df= data_df[['time']+[col for col in data_df if col != 'time']]
 #   df = df[ ['mean'] + [ col for col in df.columns if col != 'mean' ] ]
    return data_df

            
df = data_from_api(indlist)
df
df.to_csv(r'G:\MScProject\data\raw_data.csv')

print("{:=^50s}".format("Split Line"))

#part3
#create new features derived from raw data
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

df["per_1_d"] = df['price_usd_close'].pct_change(1)
df["per_3_d"] = df['price_usd_close'].pct_change(3)
df["per_7_d"] = df['price_usd_close'].pct_change(7)
df["per_15_d"] = df['price_usd_close'].pct_change(15)
df["per_30_d"] = df['price_usd_close'].pct_change(30)
df['price_back_1_d'] = df['price_usd_close'].shift(periods=1)
df['price_back_3_d'] = df['price_usd_close'].shift(periods=3)
df['price_back_7_d'] = df['price_usd_close'].shift(periods=7)
df['price_back_15_d'] = df['price_usd_close'].shift(periods=15)
df['price_back_30_d'] = df['price_usd_close'].shift(periods=30)
df['SMA_3'] = df['price_usd_close'].rolling(window=3).mean()
df['SMA_7'] = df['price_usd_close'].rolling(window=7).mean()
df.to_csv(r'G:\MScProject\data\raw_add_col.csv')

#calss for next dat price up or down
condition = [(abs(df["per_1_d"])<0.005),(df["per_1_d"]<=-0.005),(df["per_1_d"]>=0.005)]
values = [0,-1,1]
import numpy as np
df['up_do_1d'] = np.select(condition,values)


#part4
#data preprocess
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


#pip install until
#import until
# df['price_norm'] = (df['price_usd_close']-df['price_usd_close'].min())/(df['price_usd_close'].max()-df['price_usd_close'].min())
# df
def norm_data(df):
    for col in df.columns:
        if col != 'time':
            try:
                df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
            except:
                print(col+'this colmuns cant be calculated')

norm_data(df)
df.to_csv(r'G:\MScProject\data\raw_col_norm.csv')
#part5
#explore data
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#correlation matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
corrmat = df.iloc[:,1:35].corr().abs()
f, ax = plt.subplots(figsize=(12, 12))
corrmat
corrmat.to_csv(r'G:\MScProject\data\corrmat.csv')