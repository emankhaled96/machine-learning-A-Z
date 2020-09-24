# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 22:30:52 2020

@author: Eman
"""
# importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# using the pandas library to import the data in a dataframe
dataset = pd.read_csv('Mall_Customers.csv')
# Selecting the columns of features 
X = dataset.iloc[:,[3,4]].values

# using the elbow method to determine the optimal number of clusters 
# we determine this number using the within cluster sum of squares (wcss)
# in the KMeans class there is a method called inertia_ and this calculating the wcss for us which is the sum of the square of the distances between each cluster and its points
# here in this section the for loop is used to make fitting to number of clusters and then plot this elbow curve 
# we use the k-means++ as intialization to overcome the random intialization trap problem  
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters = i , init = 'k-means++' , n_init = 10 , max_iter = 300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying the kmeans to the dataset
kmeans = KMeans(n_clusters = 5 , init = 'k-means++' , n_init = 10 , max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# visualizing the clusters
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1] , s=100 , c='red' , label = 'Standard') 
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1] , s=100 , c='blue' , label = 'Careless') 
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1] , s=100 , c='green' , label = 'Target') 
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1] , s=100 , c='cyan' , label = 'Sensible') 
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1] , s=100 , c='magenta' , label = 'Careful') 
plt.scatter(kmeans.cluster_centers_[: , 0],kmeans.cluster_centers_[: , 1] , s=200 , c='yellow' , label = 'centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

