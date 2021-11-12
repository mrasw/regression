#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 04:46:48 2020

@author: basuki
"""

# import mglearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt


#mglearn.plots.plot_knn_regression(n_neighbors=3)

#import dataset
dataset = pd.read_csv('datadiabetfix_raw_bgl_52.csv')

#independent data
feature_cols = ['avgCo','avgCo2','avgKetone','avgHumid','avgTemp','avgVOC']
X = dataset[feature_cols]

#dependent data
y = dataset['class']

from sklearn.neighbors import KNeighborsRegressor

# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for i in range(1,4):

        
    # instantiate the model and set the number of neighbors to consider to 3
    reg = KNeighborsRegressor(n_neighbors=i)
    # fit the model using the training data and training targets
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_train)
    
    

    print("Neighbor = ", i)
    print("X shape: ",X.shape)
    print("y shape: ",y.shape)
    print("X train shape: ",X_train.shape)
    print("X test shape: ",X_test.shape)
    print("y train shape: ",y_train.shape)
    print("y test shape: ",y_test.shape)
    print("explained variance score: ",metrics.explained_variance_score(y_train, y_pred))
    print("max error: ",metrics.max_error(y_train, y_pred))
    print("mean absolute error: ",metrics.mean_absolute_error(y_train, y_pred))
    print("mean squared error: ",metrics.mean_squared_error(y_train, y_pred))
    print("R square score: ",metrics.r2_score(y_train, y_pred))
    print("\n")