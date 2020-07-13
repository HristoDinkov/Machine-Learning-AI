# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:25:35 2019

@author: MLH-Admin
"""

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

print(tf.__version__)

#import dataset

data = np.genfromtxt('data.csv', delimiter=',')

X = data[:,:38]
y = data[:,38:39]


class_names = ['Random', 'Lights', 'MusicOn', 'MusicOff', 'Nothing']

#X = X / 2.0
#y = y / 2.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(38, activation=tf.nn.relu),
    keras.layers.Dense(40, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)             

model.save('model.h5')