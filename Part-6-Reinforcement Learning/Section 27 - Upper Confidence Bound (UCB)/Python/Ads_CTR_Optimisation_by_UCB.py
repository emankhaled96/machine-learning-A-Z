# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:15:16 2020

@author: admin
"""
# importing libraries 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
# implementing the UCB
import math
N = 10000
d = 10
ads_selected = []
number_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if(number_of_selections[i] > 0):
            average_reward = sum_of_rewards[i] /number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] +1
    reward =dataset.values[n,ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] +reward
    total_reward = total_reward +reward
"""  In this section we implement UBC By 
    1. define the number of around or users (N)
    2. define numbers of Ads (d) 
    3. create list 
    4. [0] * d this means create a vector of zeros for Number_of_selections and sums_of_rewards
    5. 1st for loop to check all around or users 
    6. initializing ad and max_upper_bound 
    7. 2nd for loop to check all version of ads
    8. 1st step to calculate average_reward of each version of ads
    9. 2nd step to calculate delta 
    10. from 1st and 2nd step we calculate upper_bound
    11. to get maximum upper_bound we create max_upper_bound and compare it with upper_bound and put max val in max_upper_bound
    12. then we add (ad) index of ads to ads_selected list
    13. update Number_of_selections of ad index
    14. get reward of each ad for each round
    15. update sums_of_rewards and total_reward  
    
    result of this section 
    when we open ads_selected we note that in the last some iteration
    the values of them is 4 is considered Ad5 is the best ad for users """
# visualizing the ads 
plt.hist(ads_selected)
plt.title('Histogram of ads selected ')
plt.xlabel('Ads')
plt.ylabel('Number of each ad was selected')
plt.show()
    
