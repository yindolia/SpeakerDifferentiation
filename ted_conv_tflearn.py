#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:27:03 2017

@author: yindolia
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from makeTedData import gen_data, gen_melspec, gen_speaker, gen_batch, feed_data, cal_y

# Data loading and preprocessing
X, Y, testX, testY = feed_data(2)  
X = X.reshape([-1, 40, 40,1])
testX = testX.reshape([-1, 40, 40, 1])

Y = cal_y(Y)
testY = cal_y(testY)

# Building convolutional network
network = input_data(shape=[None, 40, 40, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=1,
           validation_set=({'input': testX}, {'target': testY}),
snapshot_step=100, show_metric=True, run_id='convnet_mnist')
