# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 23:09:56 2019

@author: Prateek
"""
#Importing the libraries
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt

#Importing the dataset
dataset= pd.read_csv('creditcard.csv')
X= dataset.iloc[:, 1:30].values
y= dataset.iloc[:, 30].values


#Splitting the dataset
from sklearn.cross_validation import train_test_split
X_train ,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#Applying Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

classifier= Sequential()

classifier.add(Dense(output_dim=15, init='uniform', activation='relu', input_dim=29))
classifier.add(Dense(output_dim=15, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=15, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the data to training set
classifier.fit(X_train, y_train, batch_size=50, nb_epoch=10)

#Predicting the test set results
y_pred= classifier.predict(X_test)
y_pred=(y_pred>0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)