import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing , model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
from matplotlib import axes
import matplotlib
matplotlib.use("TkAgg")
style.use('ggplot')

quandl.ApiConfig.api_key = "eyM6iMsd3D4eGdeoCPLw"
df = quandl.get("BSE/SENSEX", start_date="2017-02-21")
df = df[['Open', 'High', 'Low', 'Close']]
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100.0
df['PCT_changhe'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
df.fillna(-99999, inplace=True)  # fill all naN values to -99999
forecast_col = 'Close'

forecast_out = int(math.ceil(0.01 * len(df)))
df['lable'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['lable'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['lable'])

# print(len(X), len(y))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()