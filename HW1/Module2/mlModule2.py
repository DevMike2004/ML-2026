import numpy as np
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request 
import matplotlib.pyplot as plt

def load_housing_data():
    path = Path("datasets/housing/housing.csv")
    url = ''
    if not path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing_full = load_housing_data()

print(f"head of the data: \n {housing_full.head()}")

# Python seems to instantly print .info() calls instantly when called in a print method
# so i have to call it after
print(f"\n\n---------------------------------------------------\n\nhousing info: \n ")
housing_full.info()

print( "----------------------------------------------------\n")
print(housing_full["ocean_proximity"].value_counts())

print( "----------------------------------------------------\n")
print(housing_full.describe())

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing_full.hist(bins=50, figsize=(12,8))

plt.show()

def randSplitData(data, ratio, rng):
    indices = rng.permutation(len(data))
    print(rng.permutation(len(data)))
    testSize = int(len(data) * ratio)
    testIndices = indices[:testSize]
    trainIndices = indices[testSize:]
    return data.iloc[trainIndices], data.iloc[testIndices]

rng = np.random.default_rng(seed=42)
trainSet, testSet = randSplitData(housing_full, .2, rng)
print(len(trainSet))
print(len(testSet))

