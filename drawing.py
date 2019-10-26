import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('./dataset/prices_processed.csv')

prices = data['price'].values
bedrooms = data['bedrooms'].values
bathrooms = data['bathrooms'].values
sqft_living = data['sqft_living'].values
sqft_lot = data['sqft_lot'].values
floors = data['floors'].values
waterfront = data['waterfront'].values
view = data['view'].values
condition = data['condition'].values
age = data['age'].values
was_rebuilt = data['was_rebuilt'].values
sqft_living15 = data['sqft_living15'].values
sqft_lot15 = data['sqft_lot15'].values
sqft_above = data['sqft_above'].values
sqft_basement = data['sqft_basement'].values

# a = data.corr()
# plt.matshow(a)
# plt.xticks(range(a.shape[1]), a.columns, fontsize=14, rotation=90)
# plt.yticks(range(a.shape[1]), a.columns, fontsize=14)
# plt.show()

zipped = list(zip(list(prices), list(sqft_living15)))
print(zipped)

plt.scatter([t[1] for t in zipped], [t[0] for t in zipped], marker='o', s=1)
# plt.plot([t[1] for t in zipped], [t[0] for t in zipped])

# naming the x axis
plt.xlabel('sqft_living15')
# naming the y axis
plt.ylabel('prices')

# function to show the plot
plt.show()
