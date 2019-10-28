from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from typing import Tuple
import pandas_profiling
import pandas as pd
import numpy as np

data = pd.read_csv('./dataset/prices_original.csv')

data = data.drop([
    'id', 'date', 'lat', 'long', 'sqft_above', 'waterfront',
    'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
], axis=1)


def encode_categories(source_data, column_name, categories):
    view_encoder = OneHotEncoder(categories=[categories], handle_unknown='ignore', sparse=False)
    new_view: np.ndarray = view_encoder.fit_transform(source_data[column_name].values.reshape(-1, 1))
    for i, c in enumerate(view_encoder.categories_[0]):
        source_data.insert(1, f'{column_name} №{c}', new_view[:, i])
    return source_data.drop([column_name], axis=1)


data = encode_categories(data, 'view', range(5))
data = encode_categories(data, 'condition', range(1, 6))

# bedrooms = data['bedrooms']
# bathrooms = data['bathrooms']
# data = data.drop(['bedrooms', 'bathrooms'], axis=1)
# data.insert(1, 'rooms', bedrooms + bathrooms)

print([c for c in data])
print('\n\n')


def separate_prices(to_separate_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = to_separate_data[:, list(range(1, len(to_separate_data[0])))]
    y_arr = to_separate_data[:, [0]]

    return x_arr, y_arr


np_data = data.to_numpy()
train_data = np_data[:20000]
test_data = np_data[20000:]

scaler = preprocessing.StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

train_inputs, train_prices = separate_prices(train_data)
test_inputs, test_prices = separate_prices(test_data)

model = Ridge(alpha=.07, max_iter=90000000)
print(model.fit(train_inputs, train_prices))


def get_coefs(coefs, data_head):
    return '\n'.join([str({key: coefs[i]}) for i, key in enumerate(data_head)])


print('coeff_used', np.sum(model.coef_ != 0))
print('coeff_description', get_coefs(model.coef_[0], data.drop('price', axis=1).keys()))

print('score', model.score(test_inputs, test_prices))
