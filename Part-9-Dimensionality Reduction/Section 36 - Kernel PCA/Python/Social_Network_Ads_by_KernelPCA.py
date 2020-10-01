# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:41:42 2020

@author: Eman
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
# importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
# Features values
x = dataset.iloc[:,[2,3]].values
 
# Target values
y = dataset.iloc[:,4].values
# split the dataset into train and test dataset
# random state make the result be the same as the other people if it setted like them
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# Feature Scaling
# here we made the feature scaling on the whole training and test sets with the non numeric features
# in x_test we don't need to fit the data because it is already fitted  
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
# we apply kernel pca on a non linear separable dataset to make the line well fitting the dataset
# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2 , kernel ='rbf')
x_train = kpca.fit_transform(x_train)
x_test = kpca.transform(x_test)
# fitting logistic regression to the dataset
# logistic regression is a linear classifier so we will work from the linear model library
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train , y_train)

# predicting the test set results
y_pred = classifier.predict(x_test)

# making the confusion matrix
# this matrix gives us the number of the correct and the incorrect predictions
# the matrix output gives 65 , 3 , 8 , 24 
# the number of the correct predictions = 65 + 24 = 89
# the number of the incorrect predictions = 8 + 3 = 11
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualizing the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualizing the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()



