import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Convolution1D,MaxPooling1D, Conv2D, Conv3D, Activation, MaxPool2D, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.optimizers import Adam


#scaler = StandardScaler()
#scaler = MinMaxScaler()
seed = 2018
np.random.seed(seed)

#csv_file = "train_input_4data.csv"  # Dataset1
csv_file = "input3_edit.csv"
input_dims = 18
output_dims = 6
labeled_rate = 0.1
learning_rate = 1e-4       # learning_rate 1e-4
batch_size = 20

def read_data(csv_file):
    data = pd.read_csv(csv_file,header=None)
    input = data.iloc[:,0:input_dims]
    output = data.iloc[:,input_dims:input_dims+output_dims]
    return input, output



X, Y = read_data(csv_file)
X = np.array(X.values)
X = X.astype('float32')
Y = np.array(Y.values)
Y = Y.astype('float32')
#tx = scaler.fit_transform(X)
#ty = scaler.fit_transform(Y)
tx = X
ty = Y

test_x = tx[:200]
train_x = tx[200:]
testy = ty[:200]
trainy = ty[200:]
x_t = X[:200]
y_t = Y[:200]



# define base model
def baseline_model(model_summary=True):
    #create model
    model = Sequential()
    model.add(Dense(32, input_dim=input_dims, activation='relu'))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(output_dims))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate), metrics=['mean_squared_error'])
    if (model_summary):
        model.summary()
    return model

def lstm_model(model_summary=True):
    #create model
    model = Sequential()
    #model.add(LSTM(4, input_dim=input_dims, activation='relu'))
    model.add(LSTM(10, return_sequences=True, input_shape=(trainx.shape[1], trainx.shape[2])))
    model.add(LSTM(activation="sigmoid", units=6, recurrent_activation="hard_sigmoid"))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(output_dims))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate), metrics=['mean_squared_error'])
    if (model_summary):
        model.summary()
    return model

def cnn_model(model_summary=True):
    #create model
    
    model = Sequential()
    #model.add(LSTM(4, input_dim=input_dims, activation='relu'))
    model.add(Conv2D(6,3,input_shape=(trainx.shape[1],trainx.shape[2],1),data_format="channels_last"))
    model.add(Activation('relu'))
    model.add(Conv2D(1,1))
    model.add(Flatten())
    model.add(Dense(output_dims))
    """
    model = Sequential((
        # The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
        # Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
        # the input timeseries, the activation of each filter at that position.
        Convolution1D(3, 3, activation='relu', input_shape=(4,6), data_format='channels_last'),
        MaxPooling1D(),     # Downsample the output of convolution by 2X.
        Convolution1D(3, 3, activation='relu'),
        MaxPooling1D(),
        Flatten(),
        Dense(output_dims, activation='relu'),     
    ))
    """
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate), metrics=['mean_squared_error'])
    if (model_summary):
        model.summary()
    return model



def cnn_data(X,x_t):
    trainx = train_x.reshape(train_x.shape[0],3,6)
    trainx = train_x.reshape((trainx.shape[0],trainx.shape[1],trainx.shape[2],1))
    testx = test_x.reshape(test_x.shape[0],3,6)
    testx = test_x.reshape((testx.shape[0],testx.shape[1],testx.shape[2],1))
    return trainx,testx

def lstm_data(X,x_t):
    trainx = train_x.reshape(train_x.shape[0],3,6)
    testx = test_x.reshape(test_x.shape[0],3,6)
    return trainx,testx



def mean_absolute_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



#x_t, y_t = read_data("test_edit.csv")
#model = baseline_model()

trainx,testx = lstm_data(train_x,test_x)
model = lstm_model()
history = model.fit(trainx, trainy, validation_split=0.2, epochs=250, verbose=0, shuffle=False)
y_pred = model.predict(testx)
#y_pred = scaler.inverse_transform(y_pred)
#print(y_t-y_pred)
#y_t = np.array(y_t.values)
print(type(y_pred))

print(y_t[:,1])
print(y_pred[:,1])
mae = mean_absolute_error(y_t, y_pred, multioutput='raw_values')
mse = mean_squared_error(y_t, y_pred, multioutput='raw_values')
for i in range(6):
    mape = mean_absolute_percentage_error(y_t[:,i], y_pred[:,i])
    print("MAPE of " + str(i) + " is : " + str(mape))
print('MAE - ', mae)
print("MSE :",mse)
#  "MSE"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()



trainx,testx = cnn_data(train_x,test_x)
model = cnn_model()

#model.load_weights('forward_model_1.h5')
history = model.fit(trainx, trainy, validation_split=0.2, epochs=250, verbose=0, shuffle=False)
#model.save_weights('forward_model_1.h5')

"""
y_rf = model.predict(testx)
y_pred = scaler.inverse_transform(y_rf)
print(y_pred)
print(testy)
mae = mean_absolute_error(testy, y_pred, multioutput='raw_values')
print('MAE - ', mae)
mse = mean_squared_error(testy, y_pred, multioutput='raw_values')
print('MSE - ', mse)
#testx = [[1.1, 12, 97, 0.01, 40, 2],[1.1, 11, 97, 0.01, 40, 2],[1.1, 11, 97, 0.01, 40, 1]]
#testx = [[0.7, 5, 150, 0.02, 38, 3]]
trainx = scaler.fit_transform(trainx)
testx = scaler.transform(testx)
trainy = scaler.fit_transform(trainy) 
y_rf = model.predict(testx)
y_pred = scaler.inverse_transform(y_rf)
print(y_pred)
"""


y_pred = model.predict(testx)
#y_pred = scaler.inverse_transform(y_pred)
#print(y_t-y_pred)
#y_t = np.array(y_t.values)
print(type(y_pred))

print(y_t[:,1])
print(y_pred[:,1])
mae = mean_absolute_error(y_t, y_pred, multioutput='raw_values')
mse = mean_squared_error(y_t, y_pred, multioutput='raw_values')
for i in range(6):
    mape = mean_absolute_percentage_error(y_t[:,i], y_pred[:,i])
    print("MAPE of " + str(i) + " is : " + str(mape))
print('MAE - ', mae)
print("MSE :",mse)
#  "MSE"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()

