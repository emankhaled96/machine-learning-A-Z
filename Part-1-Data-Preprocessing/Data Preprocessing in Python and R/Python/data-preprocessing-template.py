# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:15:33 2020

@author: Eman
"""


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# importing dataset

dataset = pd.read_csv('Data.csv')


# Features values

x = dataset.iloc[:,:-1].values

# Target values

y = dataset.iloc[:,3].values

# Handling missing data

from sklearn.impute import SimpleImputer
imputer =  SimpleImputer(missing_values = np.nan,strategy = 'mean' )
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# category encoder

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
transform = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder="passthrough")
x = transform.fit_transform(x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# split the dataset into train and test dataset
# random state make the result be the same as the other people if it setted like them
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Feature Scaling
# here we made the feature scaling on the whole training and test sets with the non numeric features
# in x_test we don't need to fit the data because it is already fitted  
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
