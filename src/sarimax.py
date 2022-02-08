import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
import scipy.stats as sct
from sklearn import preprocessing
import math
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
import tqdm
from utils import *
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


def evaluate_sarimax_model(X_train, X_test, arima_order, seasonal_order, exogenous_var_train, exogenous_var_test, k1):
    # k1=len(Xtrain) ## k1 is the number of points we consider for the train ()
    mycolonne = exogenous_var_train.columns
    # history of Xvalues we use for predictions , we use the last k1 values
    history = [x for x in X_train]
    exog = np.array([[x for x in exogenous_var_train[elt]]
                     for elt in exogenous_var_train.columns]).T.tolist()
    exog_test = np.array([[x for x in exogenous_var_test[elt]]
                          for elt in exogenous_var_test.columns]).T.tolist()
    # make predictions
    predictions = list()
    L_error = list()
    for t, _ in enumerate(tqdm.tqdm(np.arange(0, len(X_test) // 96 + 1))):
        if t * 96 + 96 < len(X_test):
            model = SARIMAX(endog=history[-k1:], order=arima_order,
                            seasonal_order=seasonal_order, exog=exog[-k1:], enforce_stationarity=True)
            model_fit = model.fit(disp=0)
            yhat = model_fit.forecast(
                steps=96, exog=exog_test[t * 96:t * 96 + 96])
            predictions.extend(yhat.tolist())
            history.extend(X_test.values[t * 96:t * 96 + 96])
            exog.extend(exog_test[t * 96:t * 96 + t])
            L_error.append(rmse(X_test.values[t * 96:t * 96 + 96], yhat))
        else:
            return L_error, predictions


def Sarimax(target, df, arima_order, seasonal_order, k):
    exogenous, X = preprocessing_tuned(df, target)
    mask = exogenous.index < "2017-01-01 00:00:00"
    exogenous = exogenous[mask]
    mask_pred = (exogenous.index > "2016-06-01 00:00:00") & (exogenous.index <=
                                                             "2016-12-31 23:30:00")
    exog_train, exog_test = exogenous[~mask_pred], exogenous[mask_pred]
    # print(exog_train)
    mask_predx = (X.index > "2016-06-01 00:00:00") & (X.index <=
                                                      "2016-12-31 23:30:00")
    X_train, X_test = X[~mask_predx], X[mask_predx]

    L_error, preds = evaluate_sarimax_model(X_train=X_train, X_test=X_test, arima_order=arima_order, seasonal_order=seasonal_order,
                                            exogenous_var_train=exog_train, exogenous_var_test=exog_test, k1=k)
    longueur = len(preds)
    true_val = X_test[0:longueur]
    date = X_test.index[0:longueur]
    error = np.mean(L_error)

    plt.figure(figsize=(12, 8))

    plt.scatter(date, preds, c="r", label="$y_{hat}$", marker="*", alpha=0.5)
    plt.scatter(date, true_val, c="b",
                label="$y_{true}$", marker="x", alpha=0.5)
    plt.suptitle(
        f"Prediction with Sarimax Model of {target} on 2nd half of 2016 with MSE : {error}")
    #plt.title(f"trained in {training_time} s and forecasted in {forecast_time} s")
    plt.ylabel(f"{target}")
    plt.xlabel("Time")
    plt.legend()
    plt.grid()
    plt.savefig(
        f"../outputs/Prediction  with Sarimax Model of {target} on 2nd half of 2016.png")
    plt.show()
    print("RMSE moyen est de  : ", error)
    print("Les variables exogènes utilisées sont les suivantes : ", exog_train.columns)
