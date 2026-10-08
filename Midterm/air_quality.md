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
    language: python
    name: python3
---

# Midterm

## importing the data

```python
import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
air_quality = fetch_ucirepo(id=360) 

# data (as pandas dataframes) 
data = air_quality.data.features 
  
# metadata 
print(air_quality.metadata) 
  
# variable information 
print(air_quality.variables) 
```

## Scanning for Null values
```python
import numpy as np

df = data.replace(-200, np.nan)      # swap them all for NaN
print(df.isna().sum())
```


Here, I'm going through all the data and changing the -200 values to null because, as the websitestates, values = -200 in the data are null
There is quite a lot of them too
I also drop all nan columns because 

## Info digestion
```python
df = df.drop('NMHC(GT)', axis=1)
```
I dropped this column because more than 75% of the data inside of it was null
This would make imputation unreliable becuase I would have to deal with that large sum of null values, possibly skewing the results

```python
df.info()
```

```python
df.describe()
```

```python
print(df.select_dtypes(include="object").columns.tolist())
```
I just want to make sure the only columns containing strings are Data and Time

```python
dt = pd.to_datetime(df["Date"])
df["weekday"] = dt.dt.dayofweek
df["month"] = dt.dt.month
df = df.drop(columns=["Date"])
```
Above, I realized, the entirety of date may cause memorization issues in the model
I didn't want it to start memorizing certain dates of the year so I extracted the weekdays and months.
This way, it may find patterns in the time of year but not exact dates

```python
# Looking at the target value and it's data
df['CO(GT)'].describe()
```

## Binning
```python
df = df.dropna(subset=["CO(GT)"])
df["CO_class"], edges = pd.qcut(df["CO(GT)"], q=3, labels=[0, 1, 2], retbins=True)
df["CO_class"] = df["CO_class"].astype(int)

df['CO_class'].head()
```
I'm using CO(GT) as the target gas because carbon monoxide the most common pollutant from cars
It's also being binned for trainging purposes. If I left the info raw, it would try to guess the pure carbon monoxide numbers.
But with the binning, it is just guessing the level (low, moderate and high)
I had to drop the null values in the column as well


## Data Seperation
```python
gt_cols = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

X = df.drop(columns = gt_cols + ['CO_class'])
y_class = df['CO_class']
y_reg = df['CO(GT)']

```
I don't need the other gasses. They would just act as noise. But the sensors for them can provide valuable information
and not to mention, they also may pickup traces of the carbon monoxide.


## Checking skewedness
```python
X.drop('Time', axis=1).skew()
```
Because certain columns are heavily skewed, I am going to use median imputation on the numerical columns


## Create the Pipeline
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_cols = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH" ,"AH"]
cat_cols = ['Time', 'weekday', 'month']

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("cat_encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocess_pipe = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])
```
Above, i deciced to use median because certain num columns are skewed heavily.
I also chose to use most_frequent imputation because I'm just used to it and I've see consistent results so far




```python
from sklearn.model_selection import train_test_split

X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, yc_train, test_size=0.2, random_state=42, stratify=yc_train)
```

## Define the sgd model 
```python
import time
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.metrics import accuracy_score, f1_score

def run_sgd(**param):
    pipe = Pipeline([("preprocess", clone(preprocess_pipe)),
                     ("sgd_clf", SGDClassifier(max_iter=1000, tol=1e-3, random_state=42, **param))])
    start = time.time()
    pipe.fit(X_tr, y_tr)
    train_time = time.time() - start
    pred = pipe.predict(X_val)
    return {**param, "iters": pipe.named_steps["sgd_clf"].n_iter_, "time_s": round(train_time, 3),
            "accuracy": accuracy_score(y_val, pred), "f1": f1_score(y_val, pred, average="macro")}
```
This just allows me to import any number of parameters all at once
We are also using 1000 max iteration, but we will most likely not use them all
We also have a hard encoded stoppage. But I will probably change this for regularization later


## Loss function comparison
```python
loss_df = pd.DataFrame([run_sgd(loss=l) for l in ["log_loss", "hinge"]])
loss_df
```
After inputing all my known loss', we see that log_loss and hinge are the best. They have the highest F1 and accuracy
I am using log loss because it is a softmax family loss

## Alpha test
```python
best_loss = "log_loss"
alpha_df = pd.DataFrame([run_sgd(loss=best_loss, alpha=a) for a in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]])
alpha_df
```
After doing the above test for alpha variables, .001 is the best. It has the highest values for accuracy and f1

## Learning rate testing
```python
best_alpha = 1e-3
lr_df = pd.DataFrame([run_sgd(loss=best_loss, alpha=best_alpha, learning_rate=lr, eta0=e)
                      for lr in ["invscaling", "optimal"] for e in [0.001, 0.01, 0.1]])
lr_df
```
Optimal stays the same here for all values of e but outscores invscaling in everything. We will use optimal learning rate

## Test tuned model against Basic LogReg
```python
import time
from sklearn.linear_model import LogisticRegression

models = {
    "SGD (tuned)": SGDClassifier(loss="log_loss", alpha=0.001, max_iter=1000, tol=1e-3, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
}
for name, clf in models.items():
    pipe = Pipeline([("preprocess", clone(preprocess_pipe)), ("clf", clf)])
    start = time.time()
    pipe.fit(X_train, yc_train)
    train_time = time.time() - start
    pred = pipe.predict(X_test)
    print(name, "| iters:", np.max(pipe.named_steps["clf"].n_iter_), "| time:", round(train_time, 3),
          "| acc:", round(accuracy_score(yc_test, pred), 3), "| f1:", round(f1_score(yc_test, pred, average="macro"), 3))
```
Logistic Regression wins here no doubt. It has higher values of accuracy and F1

## Confusion Matrix
```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

print(classification_report(yc_test, pred, target_names=["Low", "Moderate", "High"]))

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(yc_test, pred, display_labels=["Low", "Moderate", "High"],
                                        cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix: CO Level Classification")
plt.show()
```
We get around 83% accuracy here. But the weakspot is definitely the moderate guesses. They were abot 68% accurate


## Setting up Baseline Regression and gettings is R2, RMSE and MAE values
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

lin_pipe = Pipeline([("preprocess", clone(preprocess_pipe)), ("reg", LinearRegression())])
lin_pipe.fit(X_train, yr_train)
pred = lin_pipe.predict(X_test)
print("RMSE:", round(root_mean_squared_error(yr_test, pred), 3), "| MAE:", round(mean_absolute_error(yr_test, pred), 3),
      "| R2:", round(r2_score(yr_test, pred), 3))
```
R2 here looks good. For a linear model on somewhat complex data, .857 is a good score. But all that says is the data has real meaning behind the CO rates

MAE is low (.362). It's not a bad error though

RMSE here is about 1.5 times more than MAE. Makes sense though because RMSE punishes large errors more than MAE

## Testing regularization of Lasso and Ridge
```python
alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
rows, coefs = [], {}
for name, Model in [("Ridge", Ridge), ("Lasso", Lasso)]:
    for a in alpha_values:
        pipe = Pipeline([("preprocess", clone(preprocess_pipe)), ("reg", Model(alpha=a, max_iter=10000))])
        pipe.fit(X_train, yr_train)
        pred = pipe.predict(X_test)
        c = pipe.named_steps["reg"].coef_
        coefs[(name, a)] = c
        rows.append({"model": name, "alpha": a, "RMSE": root_mean_squared_error(yr_test, pred),
                     "R2": r2_score(yr_test, pred), "zero_coefs": (c == 0).sum()})
results = pd.DataFrame(rows)
results
```
Ridge barely moves here. It stays within the same hundredth of an R2 value and zero_coefs stays 0 throughout. 
RMSE also stays within a hundredth the whole time. This makes sense as the weights will only zero out when the alpha is very large (for ridge).

Lasso is a bit different. It has 51 features and at the start, it removes 8. Though it stays close to Ridge there.
It ends up dropping all it's features and it's r2 is basically zero. Telling me it was just guessing at that point.
So for Lasso, the alpha got too high. We over regularized the model. Causing underfitting
Something to note is the R2 of lasso is still pretty strong at alpha = .1. This is when we only have 4 features left. 
This tells us only about 4 features have a majority of the pull


## Coefficient Table
```python
names = pipe.named_steps["preprocess"].get_feature_names_out()
coef_df = pd.DataFrame(coefs, index=names)
num_names = [n for n in names if n.startswith("num__")]
coef_df.loc[num_names].round(3)
```
I want to note that this is where we see the features lasso wanted to keep. These are likely the ones that have the most relevance to average CO levels

## Plotting coefficients against alpha
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name in zip(axes, ["Ridge", "Lasso"]):
    for f in num_names:
        ax.plot(alpha_values, [coef_df.loc[f, (name, a)] for a in alpha_values], marker="o", label=f)
    ax.set_xscale("log"); ax.set_xlabel("alpha"); ax.set_title(f"{name} coefficients vs alpha")
axes[0].set_ylabel("coefficient"); axes[1].legend(fontsize=8)
plt.show()
```
Here, we see a visualization of the features closing in on zero (for the most part) as alpha increases
Ridge is stable until the end where they jump around a bit. This follows our previous observations

Lasso is also consistent with our findings. It's a little more noticable here how they act though. 
We see the NMHC coefficient start highest and then jump even higher while all the other features stay 
near eachother (relatively). This is most likely because it had to absorb the lost weight of the 
features lasso decided to drop. This plot is basically telling us the NMHC sensor is the single
strongest predictor of average CO levels per hour.


## Comparing MinMaxScaler and StandardScaler
```python
from sklearn.preprocessing import MinMaxScaler

rows = []
for scaler in [StandardScaler(), MinMaxScaler()]:
    num_pipe_s = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", scaler)])
    cat_pipe_s = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", num_pipe_s, num_cols), ("cat", cat_pipe_s, cat_cols)])

    clf = Pipeline([("preprocess", clone(prep)), ("model", LogisticRegression(max_iter=1000, random_state=42))])
    clf.fit(X_train, yc_train)
    reg = Pipeline([("preprocess", clone(prep)), ("model", Lasso(alpha=0.01, max_iter=10000))])
    reg.fit(X_train, yr_train)

    rows.append({"scaler": type(scaler).__name__,
                 "clf_accuracy": accuracy_score(yc_test, clf.predict(X_test)),
                 "clf_iters": clf.named_steps["model"].n_iter_[0],
                 "lasso_R2": r2_score(yr_test, reg.predict(X_test)),
                 "lasso_zero_coefs": (reg.named_steps["model"].coef_ == 0).sum()})
pd.DataFrame(rows).round(3)
```
The two models basically tie with cose to the same R2 value and 35 v 36 coefficients zeroed
But StandardScaler is a bit more accurate
