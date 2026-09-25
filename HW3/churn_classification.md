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

# Assignment 3 Code Notebook

## Importing the data from kaggle
```python
import kagglehub

path = kagglehub.dataset_download("blastchar/telco-customer-churn")

path += '/telco.csv'

print("Path to dataset files:", path)
```


## Making datafram

```python
import pandas as pd

df = pd.read_csv(path)
```


## Info about the data frame and changing TotalCharges whitespace to numerical value

```python
df.info()

df['tenure'].head(20)
df['MonthlyCharges'].describe()
df['Contract'].describe()
df['PaymentMethod'].describe()
df['TotalCharges'].info()
df['TotalCharges'].head()

# This says no column has errors but that isn't true
print(df.isnull())

# This is changing the values inside to numeric values, specifically the whitespace
# It will be null afterwards
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# This shows the true result. Total Charges is the only column with errors
print(df.isnull().sum())

# No more null columns now
df['TotalCharges'] = df['TotalCharges'].fillna(0)
```

## Dropping irrelevant columns 
I removed gender and phone service as well because they will just act as noise
```python
X = df.drop(columns=['customerID', 'gender', 'PhoneService', 'Churn'])
y = df['Churn']

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

print(num_cols)
print(cat_cols)


```


## Pipelines for Cat and Num data frames
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDClassifier


num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent", fill_value="None")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
)

```

