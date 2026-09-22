import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as knr

dataRoot = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(dataRoot + "lifesat/lifesat.csv")

x = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

lifesat.plot(kind='scatter', grid=True, 
             x="GDP per capita (USD)", y="Life satisfaction")

plt.axis([23_500, 62_500, 4, 9])
plt.show()

model = knr(n_neighbors=3)

model.fit(x, y)

x_new = [[33_442.8]]

print(model.predict(x_new))
