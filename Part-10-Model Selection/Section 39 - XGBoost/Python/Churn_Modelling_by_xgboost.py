# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 01:39:04 2020

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


# xgboost helps us to get a high accuracy model with high speed running time
# Fitting the Xgboost to the dataset
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train , y_train)

# predicting the test set results
y_pred = classifier.predict(x_test)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying k-fold cross validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier , X=x_train ,y = y_train , cv =10)
accuracies.mean()
accuracies.std()
