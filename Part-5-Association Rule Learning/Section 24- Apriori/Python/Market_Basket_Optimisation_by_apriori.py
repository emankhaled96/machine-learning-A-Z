# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 01:12:35 2020

@author: Eman
"""
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
# importing dataset
# the dataset here has no names for the columns so we put that header = None  
dataset = pd.read_csv('Market_Basket_Optimisation.csv' , header = None)
# our model needs our dataset to be list of lists 
# the external list contains internal lists that contains the transactions of each basket (customer)
# so we need 2 for loops that is used to loop on all the dataset 
# j is specific for the columns and i is specific for rows
transactions = [] 
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# training apriori to the traning set 
# we will use the apriori function from the apyori class
from apyori import apriori
rules = apriori(transactions , min_support = 0.003 , min_confidence = 0.2 , min_lift = 3 , min_length = 2)

#visualizing the rules
results = list(rules)