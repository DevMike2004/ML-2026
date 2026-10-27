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

# Assignment 5

## data import
I am importing the data via sklean.datasets
The raw data from the website was in a format that would have been a little more 
complicated to insert as raw data in column format so I looked up the same datasets
for sklearn and I found that sklearn.datasets contains this data in a usable format
```python
from sklearn.datasets import fetch_openml

data = fetch_openml(data_id=37, as_frame=True)

X,y = data.data, data.target
```

## Checking out the Data
The head shows some rows of numerical columns that contain values of 0
This makes no sense for certain parts of the data like insulin and bmi where it's impossible for them to be 0. If they were, the person would be dead

```python
X.info()
```

```python
X.head()
```

```python
X.describe()
```

## Removing zero's in columns that make no sense
Here, we are going through the columns that would be impossible for it to be 0 and changing them to np.nan so the imputer knows to impute on them

```python
import numpy as np

columns = ['skin', 'insu', 'plas', 'mass', 'pres']

for col in columns:
    X[col] = X[col].replace(0, np.nan)

X.describe()

X.isna().sum()
```

## Checking the target df for missing values

all good here. just checking for values that wouldn't make sense
I also am going to turn the data in the target column into an integer
```python
y.describe()
y.head()
```
```python
y = (y == 'tested_positive').astype(int)
y.head()
```

## Pipeline definition

X data is nothing but numerical values so I don't have to split categorical and numerical data
But y is only categorical so I will still need a cat pipeline
Using median imputation because, as seen in the X.skew() cell, some of the data is heavily skewed, so mean imputation isn't good here
Otherwise, we are just using default values.


```python
X.skew()
```
```python
numCols = list(X.columns)
catCols = []

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

numPipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
catPipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessPipe = ColumnTransformer([
    ('num', numPipe, numCols),
    ('cat', catPipe, catCols)
])
```


## Transforming the data
using the predefined pipeline and cloning it on the data
I'm cloning it so it is reusable and no data leaks between usages
```python
from sklearn.base import clone
pipeline = clone(preprocessPipe)

trans = pipeline.fit_transform(X)
```

## Defining model pipelines
I'm just using default values here. 
also fitting the data to the pipes and creating a dictionary that stores them so I can just iterate through it later to get the results
```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

logRegPipe = Pipeline([
    
    ('preprocess', clone(preprocessPipe)),
    ('logreg', LogisticRegression(max_iter=1000))
])

svcPipe = Pipeline([
    ('preprocess', clone(preprocessPipe)),
    ('svc', SVC(kernel='rbf', probability=True, random_state=42))
])

treePipe = Pipeline([
    ('preprocess', clone(preprocessPipe)),
    ('tree', DecisionTreeClassifier(max_depth=5, random_state=42))
])

# fit the models
svcPipe.fit(Xtrain, yTrain)
treePipe.fit(Xtrain, yTrain)
logRegPipe.fit(Xtrain,yTrain)

# define a dictionary for the two models so I can just iterate through them later for evaluation
models = {"SVM": svcPipe, "Tree": treePipe, "LogReg": logRegPipe}
```
