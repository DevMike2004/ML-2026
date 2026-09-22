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


## Info about the data frame

```python
df.info()

df['tenure'].head(20)
df['MonthlyCharges'].describe()
df['Contract'].describe()
df['PaymentMethod'].describe()
```

## Dropping irelevant columns
I removed gender and phone service as well because they will just act as noise
```python
df = df.drop('customerID', axis=1)
df = df.drop('gender', axis=1)
df = df.drop('PhoneService', axis=1)
```

## Imputation
```python
from sklearn.impute import SimpleImputer

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.isnull().sum()
```
