# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 03:55:58 2020

@author: eman
"""
# there is no need to import the dataset because we have imported it manually
# importing the libraries
import tensorflow as tf
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential



# intializing the CNN
classifier = Sequential()

# step 1: adding the convolutional layer

classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))

#step 2: Max-Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))
# adding the second convolutional layer
classifier.add(Convolution2D(32,3,3,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#step 3: Flatten
classifier.add(Flatten())

# step 4: Full Connection
# adding the hidden layers
classifier.add(Dense(units =128 , activation = 'relu')) 
# adding the output layer
classifier.add(Dense(units = 1 , activation = 'sigmoid')) 

# Compiling the CNN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])


# Fitting the CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
# steps_per_epoch = number of images in the training set
# validation_steps = number of images in test set
classifier.fit_generator(training_set,
                    steps_per_epoch = 8000,
                    epochs = 25,
                    validation_data = test_set,
                    validation_steps = 2000)