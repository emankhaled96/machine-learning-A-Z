# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:06:44 2020

@author: eman
"""

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# importing dataset

dataset = pd.read_csv('50_Startups.csv')


# Features values

X = dataset.iloc[:,:-1].values

# Target values

Y = dataset.iloc[:,4].values

# category encoder

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
transform = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder="passthrough")
X = transform.fit_transform(X)

# avoiding the dummy variable trap
# there are some libraries that take care of dummy variable trap
# to avoid it we should exclude one of the dummy variable

X=X[:,1:]

# split the dataset into train and test dataset
# random state make the result be the same as the other people if it setted like them
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

# here we don't need feature scalig because the library will tke care of scalig

#fitting the multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
# predicting the test set

Y_pred = regressor.predict(X_test)
 

# building the optimal model using backward elimination
# to put vector of ones in the begining of matrix X bec. the coefficient of b0 = 1
# here we need to know the most effective columns and remove the non effective columns (redundant ones )
# this happens by some ways auch as backward elimination
# first we put all the columns of the independant set (X)
# then we fit the regressor to it
# here we use the ordinaryLeastSquares to make the regression
# after that we need to call the summary function that gives us a table with some useful information in it
# in this table we find avalue which is called P this P will make us know whether this column of features is effective or not
# we chose a signficant level SL = 0.05
# so we compare the highest P values among all the columns with this SL value
# if the highest P is > SL which = 0.05 then we will remove this column and repeat again to fit the regression 
# we keep making this process until all columns be with P values < SL
import statsmodels.api as sm

X = np.append(arr = np.ones((50,1)).astype(int),values = X, axis = 1)
# first trial
"""

X_opt =np.array(X[:,[0,1,2,3,4,5]], dtype=float)
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
regressor_OLS.summary()"""
# second trial
"""
X_opt =np.array(X[:,[0,1,3,4,5]], dtype=float)
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
regressor_OLS.summary()
"""
#third trial
"""
X_opt =np.array(X[:,[0,3,4,5]], dtype=float)
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
regressor_OLS.summary()
"""
# forth trial
"""
X_opt =np.array(X[:,[0,3,5]], dtype=float)
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
regressor_OLS.summary()"""
# fifth one 
X_opt =np.array(X[:,[0,3]], dtype=float)
regressor_OLS = sm.OLS(endog = Y , exog = X_opt).fit()
regressor_OLS.summary()
# after making this one we found that the most effective columns in the profit value is the constant and the R&D spends column
# they have p<0.05
