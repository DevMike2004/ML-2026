---
jupyter:
  jupytext:
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="e40101be" -->
# Assignment 3 Code Notebook
<!-- #endregion -->

```python id="a74756ad"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend',  fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

<!-- #region id="56e3d5eb" -->
## Importing the data from kaggle
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="d76ee159" outputId="15df832f-b17f-4e4c-eed7-a70161292e2a"
from pathlib import Path
import kagglehub

path = kagglehub.dataset_download("blastchar/telco-customer-churn")
print("Path to dataset files:", path)

df = pd.read_csv(Path(path) / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

<!-- #region id="870189e4" -->
## Info about the data frame
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="3e37db12" outputId="c4e5f2e7-f920-4bb1-8bd3-17af3beda18d"
df.info()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 677} id="8502856d" outputId="0aba9fdc-e33c-43aa-f052-4b27a5182649"
df[['tenure', 'TotalCharges', 'MonthlyCharges', 'Contract', 'PaymentMethod']].head(20)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 300} id="84bdf6fc" outputId="834a97f9-a79f-4616-a582-376fbf02bf52"
# TotalCharges is missing from this summary because it was loaded as text, not numbers
df.describe()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 210} id="799a8405" outputId="34ca3f34-bece-4582-c7f0-f054ab530a20"
df['Contract'].value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 241} id="b25ac82f" outputId="dbbc29f0-c9e2-4b8f-8d91-ee28731f4079"
df['PaymentMethod'].value_counts()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 178} id="7cbc239e" outputId="1db7adf1-460b-4b38-b40d-fe3cd6127764"
df['Churn'].value_counts()
```

<!-- #region id="4afcfca1" -->
## Finding the missing values in TotalCharges
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="1582b819" outputId="8827050d-d34c-4120-e21f-4394b11c49a5"
# This says no column has missing values, but that isn't true:
# the blanks in TotalCharges are spaces (" "), which count as normal text
print(df.isnull().sum())
```

```python colab={"base_uri": "https://localhost:8080/"} id="ada7afb9" outputId="2c7c3395-431a-41e6-9ed1-445717830d4a"
# Changing the values to numbers. The whitespace can't be converted,
# so errors='coerce' turns it into NaN (a real missing value)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# This shows the true result. TotalCharges is the only column with missing values
print(df.isnull().sum())
```

```python colab={"base_uri": "https://localhost:8080/", "height": 394} id="f844327b" outputId="c8520833-1a35-4924-e73e-0ffb41440cb5"
# All the missing rows have tenure 0 (new customers who haven't been billed yet)
df[df['TotalCharges'].isnull()][['tenure', 'MonthlyCharges', 'TotalCharges']]
```

<!-- #region id="bc139c7b" -->
## Dropping irrelevant columns
I removed gender and phone service as well because they will just act as noise
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="f0adbe1f" outputId="ea6d008d-a501-4e1e-d9d6-39ec3ba074c5"
X = df.drop(columns=['customerID', 'gender', 'PhoneService', 'Churn'])
df['ChurnFlag'] = (df["Churn"] == "Yes")
y = df['ChurnFlag']

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

print(num_cols)
print(cat_cols)
```

<!-- #region id="e93d4ddc" -->
## Pipelines for Cat and Num data frames
<!-- #endregion -->

```python id="a8b791bc"
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# The median imputer fills the missing TotalCharges values
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("cat_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess_pipe = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} id="ed6c8522" outputId="921c6223-2f77-4543-c8b7-d45f97845073"
from sklearn.linear_model import SGDClassifier

final_pipe = Pipeline([
    ("preprocess", preprocess_pipe),
    ("sgd_clf", SGDClassifier(loss="log_loss", random_state=42))
])
final_pipe
```

<!-- #region id="90461d9e" -->
## Training and evaluating on 3 train/test splits
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ea94f8ac" outputId="00912978-55e3-4de5-a97b-ca8653ce2e97"
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import (confusion_matrix, classification_report, precision_score,
                            recall_score, f1_score, roc_curve, roc_auc_score)

test_sizes = (0.2, 0.3, 0.4)
split_names = ("80/20", "70/30", "60/40")  # train/test

yTests, yPreds, yScoresAll = [], [], []

for test_size, split_name in zip(test_sizes, split_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=42)
    model = clone(final_pipe)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]  # probability of churning

    yTests.append(y_test)
    yPreds.append(y_pred)
    yScoresAll.append(y_scores)

    print(f"===== Split {split_name}: {len(X_train)} train, {len(X_test)} test =====")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 382} id="428a85e1" outputId="3dcf2ce5-3a77-4202-bb38-5f65502f986d"
from sklearn.metrics import ConfusionMatrixDisplay

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
plt.rc('font', size=10)
for idx, split_name in enumerate(split_names):
    ConfusionMatrixDisplay.from_predictions(yTests[idx], yPreds[idx], ax=axs[idx])
    axs[idx].set_title(f"Split {split_name}")
plt.show()
plt.rc('font', size=14)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 475} id="50467338" outputId="acc4ebbb-9875-431e-c172-1be86c8ba996"
plt.figure(figsize=(6, 5))
for idx, split_name in enumerate(split_names):
    fpr, tpr, thresholds = roc_curve(yTests[idx], yScoresAll[idx])
    auc = roc_auc_score(yTests[idx], yScoresAll[idx])
    plt.plot(fpr, tpr, linewidth=2, label=f"{split_name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=11)
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="f7fa9ed9" outputId="4b5257c2-122d-4df4-faed-5cb07a203fb3"
for idx, split_name in enumerate(split_names):
    yTest, yPred, yScores = yTests[idx], yPreds[idx], yScoresAll[idx]
    print(f"{split_name}  "
          f"accuracy: {(yPred == yTest).mean():.2%}  "
          f"precision: {precision_score(yTest, yPred):.2%}  "
          f"recall: {recall_score(yTest, yPred):.2%}  "
          f"f1: {f1_score(yTest, yPred):.2%}  "
          f"ROC-AUC: {roc_auc_score(yTest, yScores):.3f}")
```

<!-- #region id="2b05309f" -->
## Comparing StandardScaler with MinMaxScaler
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="9602d82f" outputId="fb18c394-4801-48bb-b2c2-1e5b5766ae86"
from sklearn.preprocessing import MinMaxScaler

minmax_num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler()),
])
minmax_pipe = Pipeline([
    ("preprocess", ColumnTransformer([
        ("num", minmax_num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])),
    ("sgd_clf", SGDClassifier(loss="log_loss", random_state=42))
])

for test_size, split_name in zip(test_sizes, split_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=42)
    model = clone(minmax_pipe)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    print(f"MinMaxScaler {split_name}  "
          f"accuracy: {(y_pred == y_test).mean():.2%}  "
          f"recall: {recall_score(y_test, y_pred):.2%}  "
          f"ROC-AUC: {roc_auc_score(y_test, y_scores):.3f}")
```

<!-- #region id="8c860699" -->
## 2D decision boundary (tenure vs MonthlyCharges)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 80} id="b4fed9e4" outputId="ff242f78-dff4-481d-ada2-7050e2379e4f"
X_2d = df[["tenure", "MonthlyCharges"]].values
y_2d = df["ChurnFlag"].values
X_train, X_test, y_train, y_test = train_test_split(X_2d, y_2d, test_size=0.2,
                                                    random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

sgd_2d = SGDClassifier(loss="log_loss", random_state=42)
sgd_2d.fit(X_train_scaled, y_train)
```

```python colab={"base_uri": "https://localhost:8080/"} id="81b892cf" outputId="afe2a346-deb0-4440-f406-d1136e0be0f4"
# coefficients for [tenure, MonthlyCharges]: positive pushes towards churn, negative towards staying
sgd_2d.intercept_, sgd_2d.coef_
```

```python colab={"base_uri": "https://localhost:8080/", "height": 552} id="fbd38cb8" outputId="824a6cc3-a5f3-4e11-ca35-19ad84dd8368"
from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(["#9898ff", "#a0faa0"])  # blue = stay, green = churn

# for the contour plot
x0, x1 = np.meshgrid(np.linspace(0, 72, 500).reshape(-1, 1),
                     np.linspace(15, 120, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]  # one instance per point on the figure
X_new_scaled = scaler.transform(X_new)

y_proba = sgd_2d.predict_proba(X_new_scaled)
y_predict = sgd_2d.predict(X_new_scaled)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

X_show, y_show = X_test[:500], y_test[:500]  # first 500 test customers, for readability

plt.figure(figsize=(10, 6))
plt.plot(X_show[y_show == 0, 0], X_show[y_show == 0, 1], "bs", label="Stayed")
plt.plot(X_show[y_show == 1, 0], X_show[y_show == 1, 1], "g^", label="Churned")

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap="hot")
plt.clabel(contour, inline=1)
plt.xlabel("tenure (months)")
plt.ylabel("MonthlyCharges")
plt.legend(loc="lower right")
plt.axis([0, 72, 15, 120])
plt.grid()
plt.show()
```
