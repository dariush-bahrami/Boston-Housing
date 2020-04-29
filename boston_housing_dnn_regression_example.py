# -*- coding: utf-8 -*-
import numpy as np

from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization

from path_data import model_path

# Loading Data, 404 training cases and 102 validation cases
(x_train, y_train), (x_valid, y_valid) = boston_housing.load_data()
# we have 13 predictor variables related to building age, mean number of rooms
# crime rate, the local student-to-teacher ratio, and so on.
# The median house price (in thousands of dollars) for each area is provided
# in the y variables.

# Regression model network architecture
model = Sequential()

model.add(Dense(32, input_dim=13, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear'))
# in the output layer we set the activation argument to linear—the option
# to go with when you’d like to predict a continuous variable, as we do when
# performing regression. The linear activation function outputs z directly so
# that the network’s ŷ can be any numeric value


# Compiling a regression model
model.compile(loss='mean_squared_error', optimizer='adam')


# Fitting a regression model
model.fit(x_train, y_train, batch_size=8, epochs=32, verbose=1,
          validation_data=(x_valid, y_valid))

# Saving model
model.save(model_path)
