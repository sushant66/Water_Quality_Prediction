#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:06:16 2020

@author: sushant
"""

#import pandas as pd
import quandl, math,datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, svm
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle  #saving classifier to avoid training again

style.use('ggplot')

"""
preprocessing -  for scaling data especially done on features
cross_validation or train_test_split  -  Training and testing of model
svm - explained afterwards
"""
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = df['Adj. High']  - df['Adj. Close']/df['Adj. Close'] * 100
df['PCT_Change'] = df['Adj. Close']  - df['Adj. Open']/df['Adj. Open'] * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
forecast_col = 'Adj. Close'            #wwhat to predict 
df.fillna(-99999, inplace = True)      #Acts as outlier
print(len(df))
forecast_out = int(math.ceil(0.01*len(df)))  #using last  days data to predict
print('Days in advance ', forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)  #days in future

X = np.array(df.drop(['label'],1))      #features
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]        #last 35 days
X = X[:-forecast_out]

df.dropna(inplace = True)
y = np.array(df['label'])  #label

print(len(X), len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) #shuffle data and assign
# clf = LinearRegression()            #n_jobs = -1 run as many jobs
# clf.fit(X_train, y_train)       #Training
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)  #dump classifier in f
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)    #loaded saved classifier
accuracy = clf.score(X_test, y_test)       #Testing
print('Accuracy ' , accuracy)
"""
3424
Days  35
3389 3389
0.9768887243350416

#Using SVM
clf = svm.SVR()     #Support Vector Regression    clf = svm.SVR(kernel = 'poly')
clf.fit(X_train, y_train)       #Training
accuracy = clf.score(X_test, y_test)       #Testing
print(accuracy)     #0.8481626895398251
"""
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)     #predicted 35 days unknown values
df['Forecast'] = np.nan

last_date = df.iloc[-1].name   #last value
last_unix = last_date.timestamp()
one_day = 86400         #seconds in one day
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]  #df.loc gives index 
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)           #lower right legend
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()