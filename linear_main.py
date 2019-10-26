from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from typing import Tuple
import pandas as pd
import numpy as np

data = pd.read_csv('./dataset/prices_original.csv')

data = data.drop([
    'id', 'date', 'lat', 'long',
    'yr_built', 'yr_renovated', 'sqft_lot', 'zipcode',
    # 'view', 'floors', 'waterfront', 'condition', 'premium'
], axis=1)

bedrooms = data['bedrooms']
bathrooms = data['bathrooms']

data = data.drop(['bedrooms', 'bathrooms'], axis=1)
data.insert(1, 'rooms', bedrooms + bathrooms)

print(data.axes[1])


def separate_prices(to_separate_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = to_separate_data[:, list(range(1, len(to_separate_data[0])))]
    y_arr = to_separate_data[:, [0]]

    ones = np.ones([x_arr.shape[0], 1])
    return np.concatenate((ones, x_arr), axis=1), y_arr


np_data = data.to_numpy()
train_data = np_data[:19000]
test_data = np_data[19000:]

scaler = preprocessing.StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

train_inputs, train_prices = separate_prices(train_data)
test_inputs, test_prices = separate_prices(test_data)

model = LinearRegression()
model.fit(train_inputs, train_prices)

print('score', model.score(test_inputs, test_prices))
