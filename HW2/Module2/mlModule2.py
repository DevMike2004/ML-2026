import numpy as np
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request 
import matplotlib.pyplot as plt
from zlib import crc32
from scipy.stats import binom

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

def assignIdToSet(ident, testRatio):
    # using a scalar object from the crc32 lib to find the data points in the test set
    # this also makes it so certain indices will always be in a certain set no 
    # matter if the original array size changes. This is row dependent
    # Could cause data to split in the data that is less than ideal though
    return crc32(np.int64(ident)) < testRatio * 2**32

## this is actually splitting the date that we assigned above
def splitDataHash(data, testRatio, idColumn):
    ids = data[idColumn]
    inTestSet = ids.apply(lambda id_: assignIdToSet(id_, testRatio))
    # return the dataframes containing the data
    return data.loc[~inTestSet], data.loc[inTestSet]

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

# setting up the tables and their formatting
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing_full.hist(bins=50, figsize=(12,8))

plt.show()

print("\n_______________________________________________________________________\n\n")

housingWithId = housing_full.reset_index()
trainSet, testSet = splitDataHash(housingWithId, 0.1661, "index")
print(len(testSet) / len(trainSet))

print("\n_______________________________________________________________________\n\n")

housingWithId["id"] = (housing_full["longitude"]) * 1000 + housing_full["latitude"]
trainSet, testSet = splitDataHash(housingWithId, .16, "id")
print(len(testSet) / len(trainSet))

print(testSet["total_bedrooms"].isnull().sum())
print(trainSet["total_bedrooms"].isnull().sum())

 using a binomial distribution algorithm to find the chances that sample is bad
sampleSize = 1000
femaleRatio = .516
probTooSmall = binom(sampleSize, femaleRatio).cdf(490 - 1)
probTooLarge = 1 - binom(sampleSize, femaleRatio).cdf(540)

print("\n_______________________________________________________________________\n\n")

print(probTooSmall + probTooLarge)

housing_full["income_cat"] = pd.cut(housing_full["median_income"], 
                                    bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                                    labels=[1, 2, 3, 4, 5])
catCounts = housing_full["income_cat"].value_counts().sort_index()
catCounts.plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of Districts")

plt.show()
