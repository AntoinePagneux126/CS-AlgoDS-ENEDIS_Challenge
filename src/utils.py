import pandas as pd 
import math
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer,QuantileTransformer
import scipy.stats as sct
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX



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

def keep_relevant_features(df):
    targets = ['RES1_BASE', 'RES11_BASE','PRO1_BASE', 'RES2_HC', 'RES2_HP', 'PRO2_HC', 'PRO2_HP']
    myfeatures=df.columns
    correlation=df.corr()
    colonnes=df.columns
    for ligne in colonnes:
        for col in colonnes:
            if 0.9<abs(correlation[ligne][col])<1:
                if col in myfeatures and col not in targets:
                    myfeatures=myfeatures.drop(col)
            elif correlation[ligne][col]==1:
                break
    for col in df.columns:
        if col not in myfeatures: 
            df=df.drop(columns=[col])
    return df
    
def symetric(df_reduced):
    list_to_box=[]
    list_quantile=[]
    targets = ['RES1_BASE', 'RES11_BASE','PRO1_BASE', 'RES2_HC', 'RES2_HP', 'PRO2_HC', 'PRO2_HP']
    for feature in df_reduced.columns.drop(targets):
        skew=sct.skew(df_reduced[feature])
        if abs(skew)>1:
            list_to_box.append(feature)
        else:
            list_quantile.append(feature)
    for elt in list_to_box:
        yj = PowerTransformer(method='yeo-johnson')
        data = np.array(df_reduced[elt])
        reshaped_data = np.array(data).reshape(-1, 1)
        yj.fit(reshaped_data)
        df_reduced[elt] = yj.transform(reshaped_data)
        
    rng = np.random.RandomState(304)  
    for elt in list_quantile:
        qt = QuantileTransformer(output_distribution='normal',random_state=rng)
        data = np.array(df_reduced[elt])
        reshaped_data = np.array(data).reshape(-1, 1)
        qt.fit(reshaped_data)
        df_reduced[elt] = qt.transform(reshaped_data)
    return df_reduced

def encodage(df):
    
    df_indexed=df.set_index("Horodate_UTC")
    df_indexed.index = pd.to_datetime(df.set_index("Horodate_UTC").index)
    return df_indexed


def feature_engineering(df_indexed):
    df_indexed.drop(["Mois","IDS","Horodate"],inplace=True,axis=1)
    df_indexed = keep_relevant_features(df_indexed)
    df_indexed = symetric(df_indexed)
    ss= preprocessing.StandardScaler()
    targets = ['RES1_BASE', 'RES11_BASE','PRO1_BASE', 'RES2_HC', 'RES2_HP', 'PRO2_HC', 'PRO2_HP']
    columns_targ = df_indexed.columns.drop(targets)
    df_indexed[columns_targ] = ss.fit_transform(df_indexed[columns_targ])
    df_indexed["Year"]=df_indexed.index.year
    df_indexed["Month"]=df_indexed.index.month.map(lambda x : np.cos(x*2*np.pi/12))
    df_indexed["Day"]=df_indexed.index.day.map(lambda x : np.cos(x*2*np.pi/31))
    df_indexed["Week_day"]=df_indexed.index.weekday.map(lambda x : np.cos(x*2*np.pi/7))
    df_indexed["Hour_X"],df_indexed["Hour_Y"]=zip(*pd.Series(df_indexed.index.hour).apply(getxy))
    return df_indexed

def imputation(df):
    # df.dropna(thresh=len(df)*0.9,axis=1,inplace=True)
    # df.dropna(thresh=len(df)*0.5,axis=0,inplace=True)
    return df


def preprocessing_tuned(df):
    df = encodage(df)
    df = feature_engineering(df)
    df = imputation(df)
    targets = ['RES1_BASE', 'RES11_BASE','PRO1_BASE', 'RES2_HC', 'RES2_HP', 'PRO2_HC', 'PRO2_HP']
    X = df.drop(targets, axis=1)
    y = df[targets]
    return X, y

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_arimax_model(X_train, X_test, arima_order, exogenous_var_train, exogenous_var_test):
 
    mycolonne=exogenous_var_train.columns
    history = [x for x in X_train]
    exog=np.array([[x for x in exogenous_var_train[elt]] for elt in exogenous_var_train.columns]).T.tolist()
    exog_test=np.array([[x for x in exogenous_var_test[elt]] for elt in exogenous_var_test.columns]).T.tolist()
    # make predictions
    predictions = list()

    for t in range(len(X_test)):
        model = SARIMAX(endog=history, order=arima_order, exog=exog,enforce_stationarity=False)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast(steps=1,exog=[exog_test[t]])
        predictions.append(yhat[0])
        history.append(X_test.values[t])
        exog.append(exog_test[t])
    error = rmse(X_test, predictions)

    return error, predictions

def arimax_grid_search(X_train, X_test, p_values, d_values, q_values, exogenous_var_train, exogenous_var_test):


    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse, _ = evaluate_arimax_model(
                        X_train, X_test, order, exogenous_var_train, exogenous_var_test)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print("ARIMAX(%d,%d,%d) RMSE=%.3f Exogenous =" %
                          (p, d, q, rmse))

                except:
                    continue

    print("Best ARIMAX%s MSE=%.3f" % (best_cfg, best_score))

    return best_cfg, best_score