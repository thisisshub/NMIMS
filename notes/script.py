import pandas as pd
import  quandl

import math
import numpy as np
from sklearn import preprocessing , model_selection
from sklearn.linear_model import LinearRegression

def get_prediction(stock):
    quandl.ApiConfig.api_key = "eyM6iMsd3D4eGdeoCPLw"
    df = quandl.get(stock,start_date="2017-02-21")
    df = df[['Open','High','Low','Close']]
    df['HL_PCT'] = (df['High']-df['Close'])/df['Close']*100.0
    df['PCT_changhe'] = (df['Close']-df['Open'])/df['Open']*100.0
    df.fillna(-99999, inplace=True)  # fill all naN values to -99999
    forecast_col = 'Close'

    forecast_out = int(math.ceil(0.01*len(df)))
    df['lable'] = df[forecast_col].shift(-forecast_out)
    X = np.array(df.drop(['lable'],1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)
    y = np.array(df['lable'])

    #print(len(X), len(y))
    X_train, X_test, y_train , y_test = model_selection.train_test_split(X,y,test_size=0.2)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train,y_train)
    accuracy = clf.score(X_test,y_test)
    return accuracy,clf.predict(X_lately)