from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from typing import Tuple
import pandas_profiling
import pandas as pd
import numpy as np

data = pd.read_csv('./dataset/prices_original.csv')

data = data.drop([
    'id', 'date', 'lat', 'long', 'sqft_living',
    'yr_built', 'yr_renovated', 'waterfront'
], axis=1)


def encode_categories(source_data, column_name):
    le = preprocessing.LabelEncoder()
    le.fit(source_data[column_name])
    categories = le.classes_
    print(categories)

    view_encoder = OneHotEncoder(categories=[categories], handle_unknown='ignore', sparse=False)
    new_view: np.ndarray = view_encoder.fit_transform(source_data[column_name].values.reshape(-1, 1))
    for i, c in enumerate(view_encoder.categories_[0]):
        source_data.insert(1, f'{column_name} №{c}', new_view[:, i])
    return source_data.drop([column_name], axis=1)


data = encode_categories(data, 'view')
data = encode_categories(data, 'condition')
data = encode_categories(data, 'floors')
data = encode_categories(data, 'zipcode')

print('view' in data)

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
train_data = np_data[:18000]
test_data = np_data[18000:]

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

predicted = model.predict(test_inputs)

scaled_back_predicted = scaler.inverse_transform(np.concatenate((predicted, test_inputs), axis=1))
scaled_back_expected = scaler.inverse_transform(np.concatenate((test_prices, test_inputs), axis=1))

print('mean_absolute_error', mean_absolute_error(scaled_back_predicted, scaled_back_expected))


def print_absolute_values(count):
    to_print_test = test_inputs[:count]
    to_print_expected = test_prices[:count]

    for x, y in zip(to_print_test, to_print_expected):
        y_predicted = model.predict(x.reshape(1, -1))

        scaled_back_predicted = scaler.inverse_transform(np.concatenate((y_predicted[0], x), axis=0))[0]
        scaled_back_expected = scaler.inverse_transform(np.concatenate((y, x), axis=0))[0]

        print(int(scaled_back_predicted), '<| predicted - actual |>', int(scaled_back_expected))
