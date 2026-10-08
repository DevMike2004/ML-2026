---
jupyter:
  jupytext:
    default_lexer: python
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="4e80f0ea" -->
# Homework 2
<!-- #endregion -->

```python id="608a4002" outputId="b52d60c7-e0c7-42e0-d9cb-d23290139a17" colab={"base_uri": "https://localhost:8080/"}
import pandas as pd
import numpy as np
import kagglehub

# Download latest version
download = kagglehub.dataset_download("dnkumars/cybersecurity-intrusion-detection-dataset")

print("Path to dataset files:", download)

path = f"{download}/cybersecurity_intrusion_data.csv"

df = pd.read_csv(path)
```

```python id="22252587" outputId="151382ff-c11c-4659-a790-3f42386e54d1" colab={"base_uri": "https://localhost:8080/"}
df.info()
# dropping columns I feel are useless
df = df.drop(columns=["session_id"])
df.info()
```

```python id="a0d3aece" outputId="85cea114-8485-41a5-d8f3-b86bf6b461b8" colab={"base_uri": "https://localhost:8080/", "height": 697}
df.head(20)
```


```python id="80b12a82" outputId="8a2dc70d-3518-491d-cd48-86ec0796ca75" colab={"base_uri": "https://localhost:8080/"}
df.info()
```

```python id="c7db268b" outputId="c6a6107a-852a-4240-a1da-9bfb5ded42c3" colab={"base_uri": "https://localhost:8080/", "height": 300}
df.describe()
```


```python id="ad7a0d10" outputId="a4e24a8b-427f-4df8-c3c1-381288ca0322" colab={"base_uri": "https://localhost:8080/", "height": 175}
df.describe(include="object")
```


```python id="4a35e8a3" outputId="5db867b4-f68f-4a81-c59f-9607bac9b06f" colab={"base_uri": "https://localhost:8080/", "height": 398}
df.isna().sum() # the only column that contains null values is the encryption used column
```

<!-- #region id="6396795e" -->
## Plotting the dataframe
I change the hue of the data based on the status of attack_detected.
Because it follows a binary status, I can just change the color based on the data
of the column
<!-- #endregion -->
```python id="ff0711ef" outputId="4a6bde01-80ec-4124-804e-ef9d10332ea1" colab={"base_uri": "https://localhost:8080/", "height": 1000}
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

<!-- #region id="fb7a1a5d" -->
## Column Transformation
<!-- #endregion -->
```python id="9ac77e65" outputId="dc38de34-d4ff-403d-adbc-4e12bf3ac297" colab={"base_uri": "https://localhost:8080/"}
df = df.drop(columns=["attack_label"], errors="ignore")

X = df.drop(columns="attack_detected")
y = df["attack_detected"]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

print(num_cols)
print(cat_cols)
```


```python id="49187cd3"
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


<!-- #region id="81519fd8" -->
## Next, we are splitting the data sets into a rough 20/80 split
<!-- #endregion -->

```python id="f0953059"
from sklearn.model_selection import train_test_split

xTrain, xTest, _, _ = train_test_split(X, y, test_size = 0.2, random_state = 42)
```
<!-- #region id="db0bb972" -->
## I ignored the y train and test values because I won't be using them here
<!-- #endregion -->

```python id="f6c952dd" outputId="df5c6bdb-a681-476f-ffa5-2eed5d2f4a66" colab={"base_uri": "https://localhost:8080/"}
xTrainPrep = preprocess.fit_transform(xTrain)
xTestPrep = preprocess.transform(xTest)

print(xTrainPrep.shape)
print(xTestPrep.shape)
```


```python id="7fb06be0" outputId="fae27a41-dd00-48c7-b686-edcce22c50b7" colab={"base_uri": "https://localhost:8080/", "height": 226}
pd.DataFrame(
    xTrainPrep,
    columns=preprocess.get_feature_names_out(),
    index=xTrain.index
).head()
```
