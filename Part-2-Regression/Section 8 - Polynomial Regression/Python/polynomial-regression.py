# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 23:04:44 2020

@author: Eman
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
# fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# fitting the polynomial Regression to the dataset
# here we tried the polynomial with degree 2,3,4 and we found that the best fitting graph is with degree 4 
"""from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly,y)
"""
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly,y)
# visualizing the linear regression of the dataset
plt.scatter(x, y, color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.title('Salary vs Position Levels (linear regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# by this plot we will notice that there is a huge difference between the real salaries and the predicted salaries 
# this due to it's very hard to fit this points with a line so we need a polynomial (curve)
# visualizing the polynomial regression of the dataset
"""plt.scatter(x, y, color = 'red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color = 'blue')
plt.title('Salary vs Position Levels (Polynomial regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
"""
# the last section of code was so perfect but what if we want to predict all the values of x seprated by 0.1 in this case it will make a better prediction
# so we can make it by arange function fron numpy
x_grid = np.arange(min(x),max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color = 'blue')
plt.title('Salary vs Position Levels (Polynomial regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# in this case we have got an excellant fitting curve
# predicting a new result with linear regression

lin_reg.predict([[6.5]])

# predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

