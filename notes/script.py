import pandas as pd
import  quandl
import pandas_datareader as web

import math
import numpy as np
from sklearn import preprocessing , model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def get_prediction(stock):
    quandl.ApiConfig.api_key = "eyM6iMsd3D4eGdeoCPLw"
    df = web.DataReader('AAPL', data_source='yahoo') 
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
    accuracy = (clf.score(X_test,y_test))*100

    # new code
    # plt.figure(figsize=(16,8))
    # plt.title('Close Price History')
    # plt.plot(df['Close'])
    # plt.xlabel('Date',fontsize=18)
    # plt.ylabel('Close Price USD ($)',fontsize=18)
    # plt.show()
    return accuracy,clf.predict(X_lately)

if __name__ == "__main__":
    print(get_prediction('BSE/SENSEX'))


# #Import the libraries
# import math
# import pandas_datareader as web
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
# import matplotlib.pyplot as plt
# import matplotlib 

# matplotlib.use('tkagg') 
# plt.style.use('fivethirtyeight')

# df = web.DataReader('AAPL', data_source='yahoo') 
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date',fontsize=18)
# plt.ylabel('Close Price USD ($)',fontsize=18)
# plt.show()

# #Create a new dataframe with only the 'Close' column
# data = df.filter(['Close'])#Converting the dataframe to a numpy array
# dataset = data.values#Get /Compute the number of rows to train the model on
# training_data_len = math.ceil( len(dataset) *.8) 

# #Scale the all of the data to be values between 0 and 1 
# scaler = MinMaxScaler(feature_range=(0, 1)) 
# scaled_data = scaler.fit_transform(dataset)

# #Create the scaled training data set 
# train_data = scaled_data[0:training_data_len  , : ]#Split the data into x_train and y_train data sets
# x_train=[]
# y_train = []
# for i in range(60,len(train_data)):
#     x_train.append(train_data[i-60:i,0])
#     y_train.append(train_data[i,0])

# #Convert x_train and y_train to numpy arrays
# x_train, y_train = np.array(x_train), np.array(y_train)

# #Reshape the data into the shape accepted by the LSTM
# x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
# model.add(LSTM(units=50, return_sequences=False))
# model.add(Dense(units=25))
# model.add(Dense(units=1))

# #Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# #Train the model
# model.fit(x_train, y_train, batch_size=1, epochs=1)

# #Test data set
# test_data = scaled_data[training_data_len - 60: , : ]#Create the x_test and y_test data sets
# x_test = []
# y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
# for i in range(60,len(test_data)):
#     x_test.append(test_data[i-60:i,0])

# #Convert x_test to a numpy array 
# x_test = np.array(x_test)

# #Reshape the data into the shape accepted by the LSTM
# x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# #Getting the models predicted price values
# predictions = model.predict(x_test) 
# predictions = scaler.inverse_transform(predictions)#Undo scaling

# #Calculate/Get the value of RMSE
# rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
# rmse

# #Plot/Create the data for the graph
# train = data[:training_data_len]
# valid = data[training_data_len:]
# valid['Predictions'] = predictions#Visualize the data
# plt.figure(figsize=(16,8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

# #Get the quote
# apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')#Create a new dataframe
# new_df = apple_quote.filter(['Close'])#Get teh last 60 day closing price 
# last_60_days = new_df[-60:].values#Scale the data to be values between 0 and 1
# last_60_days_scaled = scaler.transform(last_60_days)#Create an empty list
# X_test = []#Append teh past 60 days
# X_test.append(last_60_days_scaled)#Convert the X_test data set to a numpy array
# X_test = np.array(X_test)#Reshape the data
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))#Get the predicted scaled price
# pred_price = model.predict(X_test)#undo the scaling 
# pred_price = scaler.inverse_transform(pred_price)
# print(model.accuracy())
# print(pred_price)

# #Get the quote
# apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
# print(apple_quote2['Close'])