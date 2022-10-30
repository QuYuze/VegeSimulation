from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from numpy import hstack,vstack,dstack
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt  
from datetime import datetime   
import numpy as np
import statsmodels.api as sm     
from statsmodels.tsa.stattools import adfuller  
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA  
import os
import keras

def getAnnualAverage(inputArray, n):
    avgResult = np.average(inputArray.reshape(-1, n), axis=1)
    return avgResult

def getAnnualAndFlatten(input, n):
    result = list()
    y = np.shape(input)[1]
    # print(y)
    # z = np.shape(input)[2]
    # print(z)
    for i in range(y):
        # for j in range(z):
            temp = getAnnualAverage(input[:, i], n)
            result.append(temp)
    
    result = np.stack(result)

    return result

def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset|
		if end_ix>= len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix-1, :-1], sequences[end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
def split_timeseries(x,n):
    num = int(x.shape[0]*n)
    train = x[:num,:]
    test = x[num:,:]
    return train,test

def Integrated(x):
    for d in range(int(len(x) / 100)):
        temp = np.diff(x,d)
        t = adfuller(temp)  # ADF检验
        if t[1]<0.05:
            return d
        return d

def arima_para(x):
    d = Integrated(x)
    pmax = int(len(x) / 30)    
    qmax = int(len(x) / 30)
    bic_matrix = []
    for p in range(pmax +1):
        temp= []
        for q in range(qmax+1):
            try:
                temp.append(sm.tsa.arima.ARIMA(x, order = (p, d, q)).fit().bic)
            except:
                temp.append(None)
            bic_matrix.append(temp)

    bic_matrix = pd.DataFrame(bic_matrix)
    bic_matrix.fillna(bic_matrix.max(),inplace=True)   
    p,q = bic_matrix.stack().idxmin()  
    return p,d,q 

def getAnnualAverage(inputArray, n):
    avgResult = np.average(inputArray.reshape(-1, n), axis=1)
    return avgResult


def getAnnualAndFlatten(input, n):
    result = list()
    y = np.shape(input)[1]
    # print(y)
    # z = np.shape(input)[2]
    # print(z)
    for i in range(y):
        # for j in range(z):
            temp = getAnnualAverage(input[:, i], n)
            result.append(temp)
    
    result = np.stack(result)
    return result


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("--- create new folder...  ---")
    else:
        print("---  There is this folder!  ---")

class CustomCallback(keras.callbacks.Callback):
    acc = {}
    loss = {}
    best_weights = None
    
    def __init__(self, patience=None):
        super(CustomCallback, self).__init__()
        self.patience = patience
    
    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        self.loss[epoch] = logs['loss']
        self.acc[epoch] = logs['val_loss']
    
        if self.patience and epoch > self.patience:
            # best weight if the current loss is less than epoch-patience loss. Simiarly for acc but when larger
            if self.loss[epoch] < self.loss[epoch-self.patience] and self.acc[epoch] > self.acc[epoch-self.patience]:
                self.best_weights = self.model.get_weights()
            else:
                # to stop training
                self.model.stop_training = True
                # Load the best weights
                self.model.set_weights(self.best_weights)
        else:
            # best weight are the current weights
            self.best_weights = self.model.get_weights()