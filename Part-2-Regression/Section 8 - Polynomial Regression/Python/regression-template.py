# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 00:44:05 2020

@author: eman
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# importing dataset

dataset = pd.read_csv('Position_Salaries.csv')
# Features values
x = dataset.iloc[:,1:2].values
# Target values
y = dataset.iloc[:,2].values
"""# split the dataset into train and test dataset
# random state make the result be the same as the other people if it setted like them
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
"""
# we don't split the data bescause the data is very small so we need all the data to make accurate prediction
""" Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

# we don't make feature scaling because the library will take care of the scaling
# fitting the Regression Model to the dataset

# create your regressor here


# predicting a new result with linear regression

y_pred = regressor.predict([[6.5]])

# visualizing the regression model of the dataset
plt.scatter(x, y, color = 'red')
plt.plot(x,regressor.predict(x),color = 'blue')
plt.title('Salary vs Position Levels (regression model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# visualizing the regression model of the dataset (for higher resolution and smoother curve)

x_grid = np.arange(min(x),max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title('Salary vs Position Levels (Polynomial regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




