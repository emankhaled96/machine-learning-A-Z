# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:13:56 2020

@author: admin
"""
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
# we use a tsv to indicate that the delimiter in our text between the columns is the tab 
# delimiter tells the code that the tab is the thing that separating beetwwen the columns
# quoting is a some code that idicates something based on the number of it 
# here number 3 indicates that we want to ignore the double quots "
# tsv : tab separating values 
# csv : comma separating values
dataset = pd.read_csv('Restaurant_Reviews.tsv' , delimiter = '\t' , quoting =3)
# Cleaning the texts
# re is a library used to clean the texts
# sub is a method to remove some elements and put instead another element
# '[^a-zA-Z] to remove all elements in text except letters capital and small
# ' ' tto replace the removed elements with space
# .lower to get the lower case of the letters 
#  stopwords contains list of some irrevlant words , our algorithm won't make use of these words
# .split is used to split a string to list of words
# we need this list to compare each word with the words of stopwords list
# so we make a for loop that loops over every word in review and compare it with every word of stopwords list and check if this word is found in stopwords list so it removes this word
# set() : we use it because it is faster in python to deal with set rather than to deal with list  
# stemming : is to change the word to the root of this word 
# we make this to avoid sparsity
# انا كدة ببدل كل كلمة باصلها
# to make stemming we use the PorterStemmer class from nltk.stem.porter 
# we make an object from this class called ps and then apply a method from it called stem to all the words that outputted from the for loop
# ' '.join : is to join the words in the list of review and separated by space
# we apply these cleaning codes onthe whole dataset and put it in a list called corpus
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
# CountVectorizor makes a column for each word in our corpus list and the rows is our reviews number 
# it put 1 of 0 in each cell 
# 1: indicates that this word is found in this review 
# 0 : indicates that this word is not found in this review    
# after this we make a classification model to classify if this review is liked or not that means we classify the reviews to positive and negative reviews
# creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()  
y = dataset.iloc[: , 1].values  

# in classfication we usually use naive bayes , decision tree or random forest algorithms 
# split the dataset into train and test dataset
# random state make the result be the same as the other people if it setted like them
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# fitting classifier to the dataset
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train , y_train)
# predicting the test set results
y_pred = classifier.predict(x_test)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# to konw the accuracy we sum the correct predictions and divide it by all the test set
(55+91)/200