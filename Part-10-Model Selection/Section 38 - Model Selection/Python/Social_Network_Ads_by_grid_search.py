# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 23:34:28 2020

@author: eman
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
# Features values
# features here is the age and the salary
x = dataset.iloc[:,[2,3]].values
 
# Target values
y = dataset.iloc[:,4].values
# split the dataset into train and test dataset
# random state make the result be the same as the other people if it setted like them
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# Feature Scaling
# here we made the feature scaling on the whole training and test sets with the non numeric features
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# fitting classifier to the dataset
# kernel = linear makes a linear svm classifier so there will be a line between the regions
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state = 0)
classifier.fit(x_train , y_train)
# predicting the test set results
y_pred = classifier.predict(x_test)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Advanced evaluation method
# Applying k-fold cross validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier , X=x_train ,y = y_train , cv =10)
accuracies.mean()
accuracies.std()

# Applying grid search to tune the model parameters
# here our parameters are C , kernel , gamma
# cv : the number of k fold cross validation
# n_ jobs : is to make the algorithm runs faster
# best_score_ : gives us the best accuracy we can obtain with these values of parameters
# best_params_ : gives us the best value of each parameter
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1,10,100,1000],'kernel' : ['linear']},
              {'C' : [1,10,100,1000],'kernel' : ['rbf'],'gamma' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}]
grid_search = GridSearchCV(estimator = classifier
                           , param_grid = parameters,
                           scoring = 'accuracy',
                           cv =10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train , y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


