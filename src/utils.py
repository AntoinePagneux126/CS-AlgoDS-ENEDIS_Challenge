import pandas as pd 
import math
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer,QuantileTransformer,StandardScaler
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
    """[cos sin transformation for hour]

    Args:
        hour ([int or float]): [hour timestamp]

    Returns:
        [float]: [sin and cos of hour]
    """
    x = math.sin((180 - hour * 15)/180 * 3.141)
    y = math.cos((180 - hour * 15)/180 * 3.141)
    return x, y

def keep_relevant_features(df):
    """[Keep all relevant featres]

    Args:
        df ([df.Dataframe]): [Dataframe with all features]

    Returns:
        [df.Dataframe]: [Dataframe with only relevant features]
    """
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
    """[To get symetric and more normal likely distribution]

    Args:
        df_reduced ([pd.Dataframe]): [Dataset with only relevant features]

    Returns:
        [pd.Dataframe]: [Dataset with corrected features distribution]
    """
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
    """[to encode dataset and non numeric features]

    Args:
        df ([pd.Dataframe]): [Dataset to be encoded]

    Returns:
        [pd.Dataframe]: [Encoded dataframe]
    """
    df_indexed=df.set_index("Horodate_UTC")
    df_indexed.index = pd.to_datetime(df.set_index("Horodate_UTC").index)
    return df_indexed


def feature_engineering(df_indexed):
    """[Features engineering with regroup different steps]

    Args:
        df_indexed ([pd.Dataframe]): [Dataset]

    Returns:
        [pd.Dataframe]: [Dataset with new features]
    """
    df_indexed.drop(["Mois","IDS","Horodate"],inplace=True,axis=1)
    df_indexed = keep_relevant_features(df_indexed)
    df_indexed = symetric(df_indexed)
    df_indexed=df_indexed.fillna(df_indexed.mean())
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
    """[Imputation task to remove and trait missing values]

    Args:
        df ([pd.Dataframe]): [Preprocessed Dataset ]

    Returns:
        [pd.Dataframe]: [Imputed Dataset]
    """
    # df.dropna(thresh=len(df)*0.9,axis=1,inplace=True)
    # df.dropna(thresh=len(df)*0.5,axis=0,inplace=True)
    return df


def preprocessing_tuned(df):
    """[Processing the dataset]

    Args:
        df ([pd.Dataframe]): [Dataset]

    Returns:
        [tuple]: [X,y Preprocessed Dataset with X the samples and y the targets]
    """
    df = encodage(df)
    df = feature_engineering(df)
    df = imputation(df)
    targets = ['RES1_BASE', 'RES11_BASE','PRO1_BASE', 'RES2_HC', 'RES2_HP', 'PRO2_HC', 'PRO2_HP']
    X = df.drop(targets, axis=1)
    y = df[targets]
    return X,y

def split(df):
    mask = df["ds"]<"2017-01-01 00:00:00"
    df= df[mask]
    mask_pred = (df.index > "2016-06-01 00:00:00" ) & (df.index <= "2016-12-31 23:30:00")
    return df[~mask_pred],df[mask_pred]


def rmse(y_true, y_pred):
    """[Root Mean Squared Energy]

    Args:
        y_true ([list, np.ndarray, pd.Series]): [True target]
        y_pred ([list, np.ndarray, pd.Series]): [Predicted value]

    Returns:
        [float]: [RMSE score]
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_arimax_model(X_train, X_test, arima_order, exogenous_var_train, exogenous_var_test):
    """[Evaluate ARIMAX model for a certain configuration arima order]

    Args:
        X_train ([pd.Dataframe]): [the time series to predict (train part)]
        X_test ([pd.Dataframe]): [the time series to challenge the predictor (test part)]
        arima_order ([tuple]): [(p,d,q) respectively AR order, derivation order, MA order]
        exogenous_var_train ([pd.Dataframe]): [exogenous set of data for trainning]
        exogenous_var_test ([pd.Dataframe]): [exogenous set of data for testing]

    Returns:
        [tuple]: [error and predictions]
    """
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
    """[Grid search for ARIMAX Model]

    Args:
        X_train ([pd.Dataframe]): [the time series to predict (train part)]
        X_test ([pd.Dataframe]): [the time series to challenge the predictor (test part)]
        p_values ([int]): [order value of AR part]
        d_values ([int]): [order of derivation]
        q_values ([int]): [order value of MA part]
        exogenous_var_train ([pd.Dataframe]): [exogenous set of data for trainning]
        exogenous_var_test ([pd.Dataframe]): [exogenous set of data for testing]

    Returns:
        [tuple]: [(bets config, best rmse among)those tested]
    """

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