# -*- coding: utf-8 -*-

import numpy as np

from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization

# Loading Data
(x_train, y_train),(x_valid, y_valid) = boston_housing.load_data()