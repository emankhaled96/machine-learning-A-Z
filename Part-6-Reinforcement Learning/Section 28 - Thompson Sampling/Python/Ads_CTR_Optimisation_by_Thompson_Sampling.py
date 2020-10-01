# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 02:21:48 2020

@author: admin
"""
# importing libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# implementing the Thompson sampling
import random
N = 10000
d = 10
ads_selected = []
numbers_of_reward_1 = [0] * d
numbers_of_reward_0 = [0] * d
total_reward = 0
for n in range(0,N):
    ad = 0
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_reward_1[i] + 1, numbers_of_reward_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward =dataset.values[n,ad]
    if reward == 1:
        numbers_of_reward_1[ad] = numbers_of_reward_1[ad] + 1
    else:
        numbers_of_reward_0[ad] = numbers_of_reward_0[ad] + 1
    total_reward = total_reward +reward
    
# visualizing the ads 
plt.hist(ads_selected)
plt.title('Histogram of ads selected ')
plt.xlabel('Ads')
plt.ylabel('Number of each ad was selected')
plt.show()
    

