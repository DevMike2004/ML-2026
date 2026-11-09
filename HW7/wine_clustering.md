---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# **Assignemt 7**

## **Task 1**: importing the dataset and getting summary of data
below I am importing the data and loading it as a data frame directly into my feature frame and target frame
using the `return_X_y` flag and the `as_frame` flag in the `loadwine()` import method
```python
from sklearn.datasets import load_wine
import numpy as np

X, y= load_wine(return_X_y=True, as_frame=True)

X.shape
```

Below, I am getting a statistical summary of the data using `info` and `describe`
```python
X.info()
```
```python
X.describe()
```
directly checking for null values before running through possible non null missing data
```python
X.isna().sum()
```
## Missing data
the UCI website explains says that there are no missing datapoints in this dataset
sklearn uses the same dataset as the UCI website so I trust that there is no missing data here
**Also** based on the results from the data summary of the features column, there is no suspicious 
data values in the features dataframe. So I have reason to believe that the UCI description is correct

**Features**
Quick check of the target frame to make sure everything is okay here
```python
y.describe()
```
Above appears fine to me

## Checking the skewedness of the features
```python
X.skew()
```
## Pipeline Creation
defining the pipeline

**Pipeline Attributes**
- The data present has to missing data values so there is no need for an imputer
- There is also only numerical data present so I see no need for a categorical pipeline
- I still want to use a pipeline, however, in order to prevent data leakage in any case
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

## The columns names of the data
columns = list(X.columns)

## Numerical Pipeline Definition
pipe = Pipeline([
    ('scaler', StandardScaler())
])
preprocessPipe = ColumnTransformer([
    ('num', pipe, columns)
])

pipeline = clone(preprocessPipe)
```

## Data Scaling Comparison
Below, I am completing the following task in order
```python
import matplotlib.pyplot as plt
import pandas as pd

scaled = pipeline.fit_transform(X[columns])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

names = pipeline.get_feature_names_out()

dfScaled = pd.DataFrame(scaled, columns=names)

X[columns].boxplot(ax=axes[0], rot=45, grid=False)
axes[0].set_title("Raw data")
axes[0].set_ylabel("Value")

dfScaled.boxplot(ax=axes[1], rot=45, grid=False)
axes[1].set_title("After scaler pipeline")
axes[1].set_ylabel("Scaled value")

fig.suptitle("Feature distributions: before vs. after scaling", fontsize=14)
plt.tight_layout()
plt.show()
```
