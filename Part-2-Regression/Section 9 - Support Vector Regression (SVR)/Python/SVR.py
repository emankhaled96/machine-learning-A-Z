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
y = dataset.iloc[:,2:3].values
# 2:3 because fit_transform method needs an array and we need to define y as an array contains the column number 2
# when we write 2:3 the first number is included and the second number is excluded so y contains only one column which is number 2
"""# split the dataset into train and test dataset
# random state make the result be the same as the other people if it setted like them
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
"""
# we don't split the data bescause the data is very small so we need all the data to make accurate prediction
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x= sc_x.fit_transform(x)
y= sc_y.fit_transform(y)
# we have to make feature scaling because svr didn't make feature scaling by itself
# fitting SVR to the dataset


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

# predicting a new result with SVR

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

# visualizing the SVR of the dataset
plt.scatter(x, y, color = 'red')
plt.plot(x,regressor.predict(x),color = 'blue')
plt.title('Salary vs Position Levels (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualizing the SVR of the dataset (for higher resolution and smoother curve)

x_grid = np.arange(min(x),max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title('Salary vs Position Levels (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

