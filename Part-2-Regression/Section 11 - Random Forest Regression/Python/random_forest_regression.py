# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 01:58:26 2020

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
# fitting the Random Forest to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state = 0)
regressor.fit(x,y)


# predicting a new result with Random Forest regression

y_pred = regressor.predict([[6.5]])

#  in this graph we noticed that the shape is not applicable with the decision tree intiuition 
# because decision tree : it divides the dataset into intervals and predict that the values of each interval equals the average value of all the interval 
# so from 5.5 to 6.5 it must be one constant value which is approximatally equal 150k
# so in decision tree regressor we need to make our plot with a higher resolution state 
# random forest gives us the average of predictions of several decision trees to predict a more accurate value
# random forest is non continous model so we need to plot it using higher resolution

# visualizing the Random Forest model of the dataset (for higher resolution and smoother curve)

x_grid = np.arange(min(x),max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.title('Salary vs Position Levels (Random Forest regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




