import numpy as np
from typing import Tuple
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def separate_prices(normal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_arr = normal_data[:, [0]]
    x_arr = normal_data[:, list(range(1, len(data[0])))]

    ones = np.ones([x_arr.shape[0], 1])
    return np.concatenate((ones, x_arr), axis=1), y_arr


def train(x_arr: np.ndarray, y_arr: np.ndarray):
    model = LinearRegression()
    model.fit(x_arr, y_arr)
    return model


with open('./dataset/prices_processed.csv') as file:
    # type: np.array
    data = np.array([[float(v) for v in l.split(',')] for l in file.readlines()[1:]])

    # mins = np.repeat(np.array([[np.min(d) for d in data.T]]), len(data), axis=0)
    # maxs = np.repeat(np.array([[np.max(d) for d in data.T]]), len(data), axis=0)
    # normal_data = (data - mins) / (maxs - mins)

    data = np.delete(data, 11, axis=1)
    print(data[0])

    scaler = preprocessing.StandardScaler()
    normal_data = scaler.fit_transform(data)

    inputs, prices = separate_prices(normal_data[:600])
    test_inputs, test_prices = separate_prices(normal_data[600:])

    # for i in range(20):
    #     print([int(d) for d in data[i]])
    #     print(inputs[i])
    #     print()

    trained_model = train(inputs, prices)
    predicted_prices = trained_model.predict(test_inputs)

    unnormalized_train_prices = scaler.inverse_transform(np.repeat(test_prices, 15, axis=1))
    unnormalized_predictions = scaler.inverse_transform(np.repeat(predicted_prices, 15, axis=1))

    print(trained_model.score(test_inputs, test_prices))

    for it in range(50):
        i = len(predicted_prices) - it - 1

        # print(test_inputs[i], ' - ', test_prices[i], ' - ', predicted_prices[i])
        # print(test_prices[i], ' - ', predicted_prices[i])
        print(int(unnormalized_train_prices[i][0]), ' - ', int(unnormalized_predictions[i][0]))

    # print(trained_model.coef_)
    # print(mean_squared_error(unnormalized_train_prices, unnormalized_predictions))

    # for i in range(len(test_prices)):
    #     print(abs(test_prices[i] - predicted[i]))

    # print(len(test_prices))
    # print(len(predicted))

    # print(
    #     np.absolute(test_prices - predicted)
    # )

    # print([str(float(c)) for c in trained_model.coef_])

    # X = np.array()

    # print('\n'.join([str(a) for a in raw_data]))

    # X = np.array
