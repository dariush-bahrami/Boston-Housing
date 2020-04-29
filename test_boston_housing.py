# -*- coding: utf-8 -*-
from path_data import model_path

from keras.datasets import boston_housing
from keras.models import load_model

from random import randint

_, (x_valid, y_valid) = boston_housing.load_data()

model = load_model(model_path)

test_random_index = randint(0, 102)
X = x_valid[test_random_index].reshape(1, 13)
y = y_valid[test_random_index]

y_hat = model.predict(X)

print(f'for x_valid[{test_random_index}]')
print('True answer: ', y)
print('Model Estimation: ', y_hat)
