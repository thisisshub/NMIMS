# import pandas as pd
# import  quandl
# import pandas_datareader as web

# import math
# import numpy as np
# from sklearn import preprocessing , model_selection
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

# def get_prediction(stock):
#     quandl.ApiConfig.api_key = "eyM6iMsd3D4eGdeoCPLw"
#     df = web.DataReader('AAPL', data_source='yahoo') 
#     df = df[['Open','High','Low','Close']]
#     df['HL_PCT'] = (df['High']-df['Close'])/df['Close']*100.0
#     df['PCT_changhe'] = (df['Close']-df['Open'])/df['Open']*100.0
#     df.fillna(-99999, inplace=True)  # fill all naN values to -99999
#     forecast_col = 'Close'

#     forecast_out = int(math.ceil(0.01*len(df)))
#     df['lable'] = df[forecast_col].shift(-forecast_out)
#     X = np.array(df.drop(['lable'],1))
#     X = preprocessing.scale(X)
#     X_lately = X[-forecast_out:]
#     X = X[:-forecast_out]
#     df.dropna(inplace=True)
#     y = np.array(df['lable'])

#     #print(len(X), len(y))
#     X_train, X_test, y_train , y_test = model_selection.train_test_split(X,y,test_size=0.2)

#     clf = LinearRegression(n_jobs=-1)
#     clf.fit(X_train,y_train)
#     accuracy = (clf.score(X_test,y_test))*100

#     return accuracy,clf.predict(X_lately)



#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import matplotlib 
from sklearn.linear_model import LinearRegression

matplotlib.use('tkagg') 
plt.style.use('fivethirtyeight')

df = web.DataReader('AAPL', data_source='yahoo') 
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.show()

#Create a new dataframe with only the 'Close' column
data = df.filter(['Close'])#Converting the dataframe to a numpy array
dataset = data.values#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8) 

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)

#Create the scaled training data set 
train_data = scaled_data[0:training_data_len  , : ]#Split the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data into the shape accepted by the LSTM
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Test data set
test_data = scaled_data[training_data_len - 60: , : ]#Create the x_test and y_test data sets
x_test = []
y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

#Convert x_test to a numpy array 
x_test = np.array(x_test)

#Reshape the data into the shape accepted by the LSTM
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#Getting the models predicted price values
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)#Undo scaling

#Calculate/Get the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

#Plot/Create the data for the graph
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')#Create a new dataframe
new_df = apple_quote.filter(['Close'])#Get teh last 60 day closing price 
last_60_days = new_df[-60:].values#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)#Create an empty list
X_test = []#Append teh past 60 days
X_test.append(last_60_days_scaled)#Convert the X_test data set to a numpy array
X_test = np.array(X_test)#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))#Get the predicted scaled price
pred_price = model.predict(X_test)#undo the scaling 
pred_price = scaler.inverse_transform(pred_price)
clf = LinearRegression(n_jobs=-1)
X_train, X_test, y_train , y_test = model_selection.train_test_split(X,y,test_size=0.2)
clf.fit(X_train,y_train)
accuracy = (clf.score(X_test,y_test))*100
print(accuracy)
print(pred_price)

#Get the quote
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
print(apple_quote2['Close'])




# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


# import pickle
# import numpy as np
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()

# nltk.download()

# n_nodes_hl1 = 500
# n_nodes_hl2 = 500

# n_classes = 2
# hm_data = 2000000

# batch_size = 32
# hm_epochs = 10

# x = tf.placeholder('float')
# y = tf.placeholder('float')


# current_epoch = tf.Variable(1)

# hidden_1_layer = {'f_fum':n_nodes_hl1,
#                   'weight':tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
#                   'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

# hidden_2_layer = {'f_fum':n_nodes_hl2,
#                   'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
#                   'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

# output_layer = {'f_fum':None,
#                 'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
#                 'bias':tf.Variable(tf.random_normal([n_classes])),}


# def neural_network_model(data):

#     l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
#     l1 = tf.nn.relu(l1)

#     l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
#     l2 = tf.nn.relu(l2)

#     output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']

#     return output

# saver = tf.train.Saver()

# def use_neural_network(input_data):
#     prediction = neural_network_model(x)
#     with open('notes/lexicon.pickle','rb') as f:
#         lexicon = pickle.load(f)
        
#     with tf.Session() as sess:
#         sess.run(tf.initialize_all_variables())
#         saver.restore(sess,"notes/model.ckpt")
#         current_words = word_tokenize(input_data.lower())
#         current_words = [lemmatizer.lemmatize(i) for i in current_words]
#         features = np.zeros(len(lexicon))

#         for word in current_words:
#             if word.lower() in lexicon:
#                 index_value = lexicon.index(word.lower())
#                 # OR DO +=1, test both
#                 features[index_value] += 1

#         features = np.array(list(features))
#         # pos: [1,0] , argmax: 0
#         # neg: [0,1] , argmax: 1
#         result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
#         if result[0] == 0:
#             print('Positive:',input_data)
#         elif result[0] == 1:
#             print('Negative:',input_data)

# use_neural_network("He's an idiot and a jerk.")
# use_neural_network("This was the best store i've ever seen.")


# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import matplotlib.ticker as mticker
# from mpl_finance import candlestick_ohlc
# from matplotlib import style
# import requests
# import numpy as np
# import urllib
# import quandl
# import pandas_datareader as web
# import datetime as dt
# import matplotlib

# # matplotlib.use("TkAgg")

# style.use('fivethirtyeight')
# print(plt.style.available)

# print(plt.__file__)

# MA1 = 10
# MA2 = 30


# def moving_average(values, window):
#     weights = np.repeat(1.0, window) / window
#     smas = np.convolve(values, weights, 'valid')
#     return smas


# def high_minus_low(highs, lows):
#     return highs - lows


# def bytespdate2num(fmt, encoding='utf-8'):
#     strconverter = mdates.strpdate2num(fmt)

#     def bytesconverter(b):
#         s = b.decode(encoding)
#         return strconverter(s)

#     return bytesconverter


# def graph_data(stock):
#     fig = plt.figure()
#     ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=1, colspan=1)
#     plt.title(stock)
#     plt.ylabel('H-L')
#     ax2 = plt.subplot2grid((6, 1), (1, 0), rowspan=4, colspan=1, sharex=ax1)
#     plt.ylabel('Price')
#     ax2v = ax2.twinx()

#     ax3 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
#     plt.ylabel('MAvgs')

#     #df = web.DataReader(stock, data_source='yahoo')
#     quandl.ApiConfig.api_key = "eyM6iMsd3D4eGdeoCPLw"
#     df = quandl.get(stock, start_date="2017-02-21")
#     date = np.array(df.index.to_pydatetime(), dtype=np.datetime64)
#     datep = []
#     for x in date:
#         datep.append(x)
#     date = datep
#     openp = ((df['Open']).to_string(index=False)).split('\n')
#     highp = ((df['High']).to_string(index=False)).split('\n')
#     lowp = ((df['Low']).to_string(index=False)).split('\n')
#     volume = lowp
#     closep = ((df['Close']).to_string(index=False)).split('\n')


#     openp = openp[1:]
#     highp = highp[1:]
#     lowp = lowp[1:]
#     volume = volume[1:]
#     closep = closep[1:]

#     o = []
#     h = []
#     l = []
#     v = []
#     c = []

#     for x in openp:
#         o.append(float(x))
#     for x in highp:
#         h.append(float(x))
#     for x in lowp:
#         l.append(float(x))
#     v = l
#     for x in closep:
#         c.append(float(x))

#     openp = o
#     highp = h
#     lowp = l
#     closep = c
#     volume =v

#     x = 1
#     y = len(date)
#     ohlc = []
#     append_me = datep[x],openp[x], highp[x], lowp[x], closep[x], volume[x]
#     ohlc.append(append_me)
#     x += 1

#     ma1 = moving_average(closep, MA1)
#     ma2 = moving_average(closep, MA2)
#     start = len(date[MA2 - 1:])

#     h_l = list(map(high_minus_low, highp, lowp))

#     ax1.plot_date(date[-start:], h_l[-start:], '-')
#     ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='lower'))

#     #candlestick_ohlc(ax2, ohlc[-start:], width=0.4, colorup='#77d879', colordown='#db3f3f')

#     ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune='upper'))
#     ax2.grid(True)

#     bbox_props = dict(boxstyle='round', fc='w', ec='k', lw=1)

#     ax2.annotate(str(closep[-1]), (date[-1], closep[-1]),
#                  xytext=(date[-1] + 4, closep[-1]), bbox=bbox_props)

#     ax2v.fill_between(date[-start:], 0, volume[-start:], facecolor='#0079a3', alpha=0.4)

#     ax3.plot(date[-start:], ma1[-start:], linewidth=1)
#     ax3.plot(date[-start:], ma2[-start:], linewidth=1)

#     ax3.fill_between(date[-start:], ma2[-start:], ma1[-start:],
#                      where=(ma1[-start:] < ma2[-start:]),
#                      facecolor='r', edgecolor='r', alpha=0.5)

#     ax3.fill_between(date[-start:], ma2[-start:], ma1[-start:],
#                      where=(ma1[-start:] > ma2[-start:]),
#                      facecolor='g', edgecolor='g', alpha=0.5)

#     ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))
#     ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='upper'))

#     for label in ax3.xaxis.get_ticklabels():
#         label.set_rotation(45)

#     plt.setp(ax1.get_xticklabels(), visible=False)
#     plt.setp(ax2.get_xticklabels(), visible=False)
#     plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)
    
#     # graph = plt.show()
#     import mpld3
#     mpld3.fig_to_html()


# graph_data('BSE/SENSEX')