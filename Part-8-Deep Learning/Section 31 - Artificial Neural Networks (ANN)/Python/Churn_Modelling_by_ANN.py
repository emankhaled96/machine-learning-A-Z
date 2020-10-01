# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 01:52:59 2020

@author: admin
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
# importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# Features values
x = dataset.iloc[:,3:13].values
# Target values
y = dataset.iloc[:,13].values
# category encoder
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
transform = ColumnTransformer([("1", OneHotEncoder(), [1])], remainder="passthrough")
labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])
x = transform.fit_transform(x)
x= x[:,1:]
# split the dataset into train and test dataset
# random state make the result be the same as the other people if it setted like them
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# Feature Scaling
# here we made the feature scaling on the whole training and test sets with the non numeric features
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# making the ANN Model 

# 1- importing the ANN libraries
import tensorflow as tf
# 2- intializing the ANN
classifier = tf.keras.models.Sequential()

# adding the first layer and the first hidden layer
# units  : is the number of nodes in the hidden layer which equals( no.of input layer + no. of o/p layer )/2 which is the average value
# activation : we put in it the method that we use to make the activation units
classifier.add(tf.keras.layers.Dense(units = 6 , activation = 'relu' , input_dim = 11)) 

# adding the second hidden layer
classifier.add(tf.keras.layers.Dense(units = 6 , activation = 'relu')) 

# adding output layer 
# if we want more than 1 output node then there is 2 changes will be done in this code 
# 1 - units = no. of nodes you want
# 2 - activation = softmax which is the sigmoid function but for dependant variable that deals with more than two categories
classifier.add(tf.keras.layers.Dense(units = 1 , activation = 'sigmoid' )) 

# compiling the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

# fitting the ANN to the training set
classifier.fit(x_train , y_train , batch_size = 10 , epochs = 100)
# predicting the test set results
y_pred = classifier.predict(x_test)
# putting threshold value to convert probability to true Or false
y_pred = (y_pred > 0.5)
# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


