#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 02:39:17 2022

@author: song
"""
#---------------------------------packages-------------------------------------------#
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense,Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
#----------------------------------------------------------------------------#


def scaler_data(df_for_training_scaled):
    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training_scaled)
    df_for_training_scaled = scaler.transform(df_for_training_scaled)
    #return scaled data
    return df_for_training_scaled,scaler

#----------------------------------------------------------------------------#
def refor_data(df_for_training_scaled,n_futures,n_past):
    trainX = []
    trainy = []
    for i in range (n_past, len(df_for_training_scaled) - n_futures - 1):
#         trainX.append(df_for_training_scaled[i-n_past : i, 1:2])
        trainX.append(df_for_training_scaled[i-n_past : i, 1:df_for_training_scaled.shape[1]])
        trainy.append(df_for_training_scaled[i + n_futures -1:i+n_futures,1])
        
    trainX, trainy = np.array(trainX),np.array(trainy)
    return trainX, trainy

#----------------------------------------------------------------------------#
def lstm_model(units,activation,trainX, trainy):
    model = Sequential()
    model.add(LSTM(units = units,  activation = activation,  input_shape = (trainX.shape[1],trainX.shape[2]),return_sequences = True))
#     model.add(LSTM(units = units,  activation = activation,  input_shape = (trainX.shape[1],trainX.shape[2]),return_sequences = True))
    model.add(LSTM(units = int(units/2),activation = activation,  return_sequences = False))
    #The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
    model.add(Dropout(rate=0.2))
    model.add(Dense(trainy.shape[1]))
    
    model.compile(optimizer ='adam' ,loss='mse')
    model.summary()
    return model


#----------------------------------------------------------------------------#

# def split_Xy(df_for_training_scaled,n_futures,n_past): 
#     trainX, trainy = refor_data(df_for_training_scaled,n_futures,n_past)
#     trainX, trainy = np.array(trainX),np.array(trainy)
    
#     return trainX, trainy 


#--------------------------------------fit model-------------------------------------#
def fit_model(trainX, trainy,units,activation,epochs,batch_size,validation_split):

    model = lstm_model(units=units,activation=activation,trainX=trainX, trainy=trainy)
    history = model.fit(trainX,trainy,epochs=epochs,batch_size=batch_size,validation_split=0.2,verbose =1)
    
    return model, history 


#----------------------------------------------------------------------------#
def plt_loss(history):
    #fig = plt.figure()
    his = pd.DataFrame(history.history)
    his.head(10)
    his['gap']=his['loss']-his['val_loss']
    his.sort_values(by = 'gap',ascending=0)
    
    
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()
    
    
       

#----------------------------------------------------------------------------#

def predict_model(n_past,n_days_for_prediction,df_time,model,scaler,trainX,df_for_training):
    predict_period_dates = pd.date_range(list(df_time)[-n_past], periods=n_days_for_prediction).tolist()
    prediction = model.predict(trainX[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction

    #Perform inverse transformation to rescale back to original range
    #Since we used 5 variables for transform, the inverse expects same dimensions
    #Therefore, let us copy our values 5 times and discard them after inverse transform
    prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,1]
    
    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())
        
    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
    df_forecast['Date']=pd.to_datetime(df_forecast['Date'])
    
    
    original = df[['time_real', 'price_usd_close']]
    original['time_real']=pd.to_datetime(original['time_real'])
    original = original.loc[original['time_real'] >= '2018-12-1' ].loc[original['time_real'] <= '2020-12-1' ]
    df_forecast = df_forecast.loc[df_forecast['Date'] >= '2018-12-1' ].loc[df_forecast['Date'] <= '2020-12-1' ]
    
    
    return df_forecast,original

#----------------------------------------------------------------------------#

def combine_result(df_forecast,original):
    
    compare = original.merge(df_forecast,how='outer',left_on = 'time_real',right_on='Date',sort = 'time_real')
    compare1 = compare.drop('Date',axis =1).dropna(axis = 0,how='any')
    
    fst_ind = compare1.index[0] 
    compare1['gap'] = compare1['price_usd_close']-compare1['Open']
    compare1['act_change'] = compare1['price_usd_close'].diff()
    compare1['pred_change'] = compare1['act_change']
    compare1['gap_act_err'] = compare1['act_change']
    compare1['gap_pred_err'] = compare1['act_change']
    
    for i in range(len(compare1['price_usd_close'])-1):
        compare1['pred_change'][i+fst_ind+1] = compare1['Open'][i+fst_ind+1] - compare1['price_usd_close'][i+fst_ind]
        compare1['gap_act_err'][i+fst_ind+1] = compare1['act_change'][i+fst_ind+1]/compare1['price_usd_close'][i+fst_ind]
        compare1['gap_pred_err'][i+fst_ind+1] = compare1['pred_change'][i+fst_ind+1]/compare1['price_usd_close'][i+fst_ind]

    
    import numpy as np
    #create 2 classes prediction error
    condition = [(abs(compare1["gap_act_err"])<=0.005),(compare1["gap_act_err"]< -0.005),(compare1["gap_act_err"]> 0.005)]
    values = [-1,0,1]
    compare1['gap_act_err_class_3'] = np.select(condition,values)
    
    #create 2 classes prediction error
    condition = [(abs(compare1["gap_pred_err"])<=-0.005),(compare1["gap_pred_err"]< -0.005),(compare1["gap_pred_err"]> 0.005)]
    values = [-1,0,1]
    compare1['gap_pred_err_class_3'] = np.select(condition,values)
    
    

    #create 3 classes prediction error
    condition = [(compare1["gap_act_err"]< 0),(compare1["gap_act_err"]>= 0)]
    values = [-1,1]
    compare1['gap_act_err_class_2'] = np.select(condition,values)
    
    #create 3 classes prediction error
    condition = [(compare1["gap_pred_err"]< 0),(compare1["gap_pred_err"]>= 0)]
    values = [-1,1]
    compare1['gap_pred_err_class_2'] = np.select(condition,values)
    
    
    
    compare1['mean'] = compare1['gap'].mean()
    
    
    from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score,mean_squared_error
    acc3 = accuracy_score(compare1['gap_act_err_class_3'], compare1['gap_pred_err_class_3'])
    acc2 = accuracy_score(compare1['gap_act_err_class_2'], compare1['gap_pred_err_class_2'])
    print(acc3,acc2)

    mse = mean_squared_error(compare1['price_usd_close'], compare1['Open'])
    print('real value mse is '+ str(mse))
    
    
    from sklearn import preprocessing
    mse2 = mean_squared_error(preprocessing.normalize([compare1['price_usd_close']]), preprocessing.normalize([compare1['Open']]))
    
    print('normlized data mse is '+str(mse2))
    return compare1,mse,mse2,acc3,acc2

#----------------------------------------------------------------------------#
def plt_result(compare1):
   
    sns.lineplot(x=compare1['time_real'], y=compare1['price_usd_close'])
    sns.lineplot(x=compare1['time_real'], y=compare1['Open'])
    plt.show()
    
    plt.plot(compare1['time_real'], compare1['gap'])
    plt.plot(compare1['time_real'], compare1['mean'])
    # plt.plot(compare1['gap'].mean())
    #plt.legend()
    mean = compare1['gap'].mean()
    std = compare1['gap'].std()
    print('the mean value of difference between prediction and real value is '+ str(mean))
    print('the std value of difference between prediction and real value is '+ str(std))
    
    return mean,std
    
    
    

#----------------------------------------------------------------------------#
# parameters set
def para_set():
    #units = [16,32,64,128,256]
    #activation = ['relu','tanh']
    #epochs = [10,50,100,200,400,800]
    #batch = [1,2,4,8,16,32]
    # n_futures = []
    #n_past =[1,3,5,7,15,30]
    
    units = [16,32]
    activation = ['relu']
    epochs = [10]
    batch = [16]
    # n_futures = []
    n_past =[5,7]
    num = 0
    config = list()
 
    for i in units:
        for j in activation:
            for k in epochs:
                for l in batch:
                    for m in n_past:
                        num += 1
                        cof = [num,i,j,k,l,m]
                        config.append(cof)
    print('all configs{}'.format(config))
    return config
                    

#----------------------------------------------------------------------------#








#----------------------------------------------------------------------------#
#running lstm for once with specific parameter an return the result

#global df,df_x,df_y,df_time,cols,df_training,df_for_training_scaled,scaler
#global trainX, trainy,model,history ,df_forecast,original,compare1
    
df = pd.read_csv(
            r'/Users/song/Desktop/MScProject/data/raw_add_col.csv').iloc[10:, ]
df['1_day_shift'] = df['price_usd_close'].shift(periods=1)
df_X = df.drop(columns=['time', 'time_real', 'up_down',
                       'up_do_1d', 'per_1_d', 'SMA_3', 'SMA_7', '1_day_shift'])
df_y = pd.DataFrame(df['price_usd_close'])
df_time = pd.to_datetime(df['time_real'])
cols = df_X.columns
df_for_training = df_X[cols].astype(float)
    
def run_lstm(units, activation, epochs, batch, n_past):

    result=[]
    df_for_training_scaled, scaler = scaler_data(df_for_training)

    #model-----------#
    trainX, trainy = refor_data(
        df_for_training_scaled, n_futures=1, n_past=n_past)
    model, history = fit_model(trainX, trainy, units=units, activation=activation,
                               epochs=epochs, batch_size=batch, validation_split=0.2)

    plt_loss(history)
    df_forecast, original = predict_model(n_past=1005, n_days_for_prediction=1000, df_time=df_time,
                                          model=model, scaler=scaler, trainX=trainX, df_for_training=df_for_training)
    compare1,mse,mse2,acc3,acc2 = combine_result(df_forecast, original)
    
    mean,std = plt_result(compare1)
    
    #========
    result=[units, activation, epochs, batch, n_past,mse,mse2,acc3,acc2,mean,std]
    
    return result
    
#----------------------------------------------------------------------------#
def grid_serach():
    config = para_set()
    result_lst = []
    cols_name= ['units', 'activation', 'epochs', 'batch', 'n_past','mse','mse2','acc3','acc2','mean','std']
    for i in range(len(config)):

        num ,units, activation, epochs, batch, n_past = config[i]
        print('this is the {} round'.format(i))
        print('params are units:{}, activation:{}, epoch:{}, batch:{}, n_past:{}'.format(units, activation, epochs, batch, n_past))
        result = run_lstm(units=units, activation=activation, epochs=epochs, batch=batch, n_past=n_past)
        result_lst.append(result)
        
    
    result_df = pd.DataFrame(result_lst,columns=cols_name)
    result_df.sort_values(by=['mse2','acc3'],ascending=False)
    best = result_df.sort_values(by=['mse2','acc3'],ascending=True)[0:1]
    
    print('the best result is {}'.format(best) )
    
    return result_df
                           
        
        
        
        
 
    


#----------------------------------------------------------------------------#
#2. import dataset and split dataset 
df = pd.read_csv(r'/Users/song/Desktop/MScProject/data/raw_add_col.csv').iloc[10:,]
df['1_day_shift']=df['price_usd_close'].shift(periods=1)
df_X = df.drop(columns = ['time','time_real','up_down','up_do_1d','per_1_d','SMA_3','SMA_7','1_day_shift'])
df_y = pd.DataFrame(df['price_usd_close'])

df_time = pd.to_datetime(df['time_real'])

cols = df_X.columns

df_for_training  = df_X[cols].astype(float)

import time
to = time.time()
df_for_training_scaled,scaler = scaler_data(df_for_training)
trainX, trainy = refor_data(df_for_training_scaled,n_futures=1,n_past=7)
model,history = fit_model(trainX,trainy,units=64,activation='relu',epochs=300,batch_size=32,validation_split=0.2)

plt_loss(history)
df_forecast,original = predict_model(n_past=1005,n_days_for_prediction=1000,df_time=df_time,model=model,scaler=scaler,trainX=trainX,df_for_training=df_for_training)
compare1 = combine_result(df_forecast,original)
mean,std = plt_result(compare1)
t1 = time.time()
t = t1-t0
print(t)
