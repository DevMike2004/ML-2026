---
jupyter:
  jupytext:
    default_lexer: python
    formats: ipynb,md
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

# Homework 2

```python
import pandas as pd 
import numpy as np
import kagglehub

# Download latest version
download = kagglehub.dataset_download("dnkumars/cybersecurity-intrusion-detection-dataset")

print("Path to dataset files:", download)

path = f"{download}/cybersecurity_intrusion_data.csv"

df = pd.read_csv(path)
```

```python
df.info()
# dropping columns I feel are useless
df = df.drop(columns=["session_id"])
df.info()
```

```python
df.head(20)
```


```python
df.info()
```

```python
df.describe()
```


```python
df.describe(include="object")
```


```python
df.isna().sum() # the only column that contains null values is the encryption used column
```

## Plotting the dataframe
I change the hue of the data based on the status of attack_detected. 
Because it follows a binary status, I can just change the color based on the data
of the column
```python
import seaborn as sns
import matplotlib.pyplot as plt

df["attack_label"] = df["attack_detected"].map({0: "Normal", 1: "Attack"})

num = df.select_dtypes(include="number")
plt.figure(figsize=(10, 8))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Pearson correlation — numerical features")
plt.show()

sns.histplot(data=df, x="network_packet_size", hue="attack_label", kde=True)
plt.title("Packet size by outcome")
plt.show()

sns.histplot(data=df, x="session_duration", hue="attack_label", kde=True)
plt.title("Session duration by outcome")
plt.show()

sns.scatterplot(data=df, x="session_duration", y="network_packet_size",
                hue="attack_label", alpha=0.3)
plt.title("Duration vs packet size")
plt.show()
```

## Column Transformation
```python
df = df.drop(columns=["attack_label"], errors="ignore")

X = df.drop(columns="attack_detected")
y = df["attack_detected"]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

print(num_cols)
print(cat_cols)
```


```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="None")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])
```


## Next, we are splitting the data sets into a rough 20/80 split

```python
from sklearn.model_selection import train_test_split

xTrain, xTest, _, _ = train_test_split(X, y, test_size = 0.2, random_state = 42)
```
## I ignored the y train and test values because I won't be using them here

```python
xTrainPrep = preprocess.fit_transform(xTrain)
xTestPrep = preprocess.transform(xTest)

print(xTrainPrep.shape)
print(xTestPrep.shape)
```


```python
pd.DataFrame(
    xTrainPrep,
    columns=preprocess.get_feature_names_out(),
    index=xTrain.index
).head()
```
