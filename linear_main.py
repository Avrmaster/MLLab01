from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from typing import Tuple
import pandas as pd
import numpy as np

data = pd.read_csv('./dataset/prices_original.csv')

data = data.drop([
    'id', 'date', 'lat', 'long', 'sqft_above', 'waterfront',
    'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
], axis=1)

print([c for c in data])


def encode_categories(source_data, column_name, categories):
    view_encoder = OneHotEncoder(categories=[categories], handle_unknown='ignore', sparse=False)
    new_view: np.ndarray = view_encoder.fit_transform(source_data[column_name].values.reshape(-1, 1))
    for i, c in enumerate(view_encoder.categories_[0]):
        source_data.insert(1, f'{column_name} №{c}', new_view[:, i])
    return source_data.drop([column_name], axis=1)


data = encode_categories(data, 'view', range(5))
data = encode_categories(data, 'condition', range(1, 6))
data = encode_categories(data, 'premium', range(2))

print([c for c in data])

bedrooms = data['bedrooms']
bathrooms = data['bathrooms']

data = data.drop(['bedrooms', 'bathrooms'], axis=1)
data.insert(1, 'rooms', bedrooms + bathrooms)

print('\n\n')


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

model = Lasso(alpha=.01, max_iter=6000000)
print(model.fit(train_inputs, train_prices))
print(model.coef_)

print('score', model.score(test_inputs, test_prices))
