#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:41:37 2018

@author: lukasz
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import time
tfd = tf.distributions

# Hyperparamters:
version = 1
n_samples = 50

data_dimension = 2
code_size = 2
hidden_size = 200
learning_rate=0.001
num_epochs = 40
batch_size = 32
n_batches = n_samples // batch_size



tf.reset_default_graph()

# GENERATE SAMPLE DATA
mean = [0.5, 0.5]
cov = [[1, -1], [-1, 2]]
x, y = np.random.multivariate_normal(mean, cov, n_samples).T
x = x.reshape((-1,1))
y = y.reshape((-1,1))

z = np.concatenate((x, y), axis=1)

plt.scatter(z[:,0], z[:,1])
plt.show()
N =60
g1 = [0.6 + 0.6 * np.random.rand(N), np.random.rand(N)]
g2 = (0.4+0.3 * np.random.rand(N), 0.5*np.random.rand(N))
g3 = (0.3*np.random.rand(N),0.3*np.random.rand(N))