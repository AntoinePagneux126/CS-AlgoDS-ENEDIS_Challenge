import pandas as pd 
import math


def merge(file_in="../dataset/inputs.csv",file_out="../dataset/outputs.csv"): 
    """[merge inputs and outputs datasets]

    Args:
        file_in (str, optional): [path to inputs csv file]. Defaults to "../dataset/inputs.csv".
        file_out (str, optional): [path to outputs csv file]. Defaults to "../dataset/outputs.csv".
    """
    if not isinstance(file_in,str) or not isinstance(file_out,str):
        raise Exception("file must be string which represent path to csv files")

    df_in  = pd.read_csv(file_in)
    df_out = pd.read_csv(file_out)
    df = pd.merge(df_in,df_out,how="inner",
                    on = ['IDS', 'Horodate_UTC', 'Horodate', 'Mois'],
                    sort=True,
                    copy=True,
                    indicator=False,
                    validate=None,
                )
    df.to_csv("../dataset/inout.csv",index=False)

def getxy(hour):
    x = math.sin((180 - hour * 15)/180 * 3.141)
    y = math.cos((180 - hour * 15)/180 * 3.141)
    return x, y

def encodage(df):
    return df 


def feature_engineering(df):
    df_indexed=df.set_index("Horodate_UTC")
    df_indexed.index = pd.to_datetime(df.set_index("Horodate_UTC").index)
    df_indexed["Year"]=df_indexed.index.year
    df_indexed["Month"]=df_indexed.index.month
    df_indexed["Day"]=df_indexed.index.day
    df_indexed["week_day"]=df_indexed.index.weekday
    df_indexed["Hour_X"],df_indexed["Hour_Y"]=zip(*pd.Series(df_indexed.index.hour).apply(getxy))




    return df_indexed

def imputation(df):
    return(df.dropna(axis=0))


def preprocessing(df):
    df = encodage(df)
    df = feature_engineering(df)
    df = imputation(df)
    X = df.drop('Target', axis=1)
    y = df['Target']
    print(y.value_counts())
    return X, y