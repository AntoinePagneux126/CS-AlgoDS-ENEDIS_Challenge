# %%
import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime 
import warnings 
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
warnings.simplefilter('ignore')
import math
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import scipy.stats as sct
from sklearn import preprocessing
import time as time
from utils import * 
from mailsender import send_csv_by_mail
# %%
NB_PRED = 10271

def Proph(target,df):
    X,y= preprocessing_tuned(df,target)
    mask = X.index<"2017-01-01 00:00:00"
    X= X[mask]
    regressors = X.columns
    mask_pred = (X.index > "2016-06-01 00:00:00" ) & (X.index <= "2016-12-31 23:30:00")
    X_past,_ =  X[~mask_pred],X[mask_pred]

    mydf=df[['Horodate_UTC',target]]
    mydf["Horodate_UTC"]=mydf["Horodate_UTC"].apply(change_date_format)
    mydf=mydf.rename(columns={"Horodate_UTC":"ds",target:"y"})
    print(df.shape)

    mydf_past,mydf_forecast=split(mydf,"ds")
    for regressor in regressors : 
        print(regressor)
        mydf_past[regressor] = X_past[regressor].values

    m=Prophet(interval_width=0.95)
    m.add_country_holidays(country_name='FR')
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.add_seasonality(name="season", period = 91.3125 , fourier_order=7)
    for regressor in regressors: 
        m.add_regressor(regressor)
    t_start = time.time()
    m.fit(mydf_past)
    training_time = round(time.time()-t_start,3)
    future=m.make_future_dataframe(periods=NB_PRED,freq= '30min')
    for regressor in regressors : 
        future[regressor] = X[regressor].values
    t_start = time.time()
    forecast=m.predict(future)
    forecast_time = round(time.time()-t_start,3)

    print(forecast.shape)
    print(mydf_past.shape)
    print(mydf_forecast.shape)

    y_hat=forecast["yhat"][-NB_PRED:]
    y_true = mydf_forecast["y"][-NB_PRED:]
    date = forecast["ds"][-NB_PRED:]
    error = round(rmse(y_hat,y_true),3)
    print(f"Error on {NB_PRED} points : {error} RMSE ")

    plt.figure(figsize=(12,8))
    plt.scatter(date,y_hat,c="r",label="$y_{hat}$",marker="*",alpha=0.5)
    plt.scatter(date,y_true,c="b",label="$y_{true}$",marker="x",alpha=0.5)
    plt.suptitle(f"Prediction of {target} on 2nd half of 2016 with MSE : {error}")
    plt.title(f"trained in {training_time} s and forecasted in {forecast_time} s")
    plt.ylabel(f"{target}")
    plt.xlabel("Time")
    plt.legend()
    plt.grid()
    plt.savefig(f"../outputs/Prediction of {target} on 2nd half of 2016.png")
    #forecast.to_csv(f"../outputs/{target}_by_Prophet.csv")
    try : 
        send_csv_by_mail()
    except :
        print("mail will not be sended")


