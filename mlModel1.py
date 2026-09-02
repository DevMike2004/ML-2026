# importing everything I need
# Things like pandas for the DataFrame object and sklearn for the 
# different data frame manipulation tools
# and finally, the actual data set we are using 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

def makeSpace() -> None:
    print("__________________________________\n\n")

# casting the Any type onto the DATA set to stop irrelavent errors from popping up
# (vim is retarded sometimes)
DATA = fetch_california_housing()

# Here we are assigning the data a dataframe and telling it to use the data 
# of the data set and the names of the columns for each piece of data
# we are also assigning the target column its individual data
# Finally, we print the head of the data table to get a rough understanding of 
# the data we are using
df = pd.DataFrame(DATA.data, columns=DATA.feature_names)
df['target'] = DATA.target
print(df.head())

makeSpace();

# Here, we are assigning the axies of the graph. X is all columns other than target
# axis=1 tells the variable to pull in data from all columns instead of all rows
# y is the target column
X = df.drop('target', axis=1)
y = df['target']

# this is splitting the data up between different variables
# We take .2% of the data and assign it as a test set and the rest is for training 
# We also mix up the data in sets of 42 to randomize the data so that the model 
# doesnt create false predicitons based on data trends
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.2, random_state=42)

# This actually makes the model and tells it to look at the different training 
# variables to figure out the math behind the training
model = LinearRegression()
model.fit(X_train, y_train)

# here, the model is acutally predicting the X_test target values based on the 
# data present in the X_test variable
# We then calculate the MSE based on the actual values from te y_test variable using 
# the predicitons variable we previously made
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# PRINT
print(f'MSE: {mse} | type: {type(mse)}')
makeSpace()

def logTraingingMetric(epoch: int, loss: float, optName: str) -> None:
    if loss < 0:
        print("***Warning: Negative loss, checking dataset\n")
    
    print(f"Epoch {epoch:03d} | Optimizer: {optName:12s} | Loss: {loss:.4f}")

logTraingingMetric(10, 2.5694019, "Mike")

makeSpace()

modelConf = {
    "Alorithm": "Random Forest",
    "n-estimators": 100,
    "max-depth": 5,
    "ranState":42,
}

if "ranState" in modelConf :
    print(f"Tree state is: {modelConf['ranState']}")

makeSpace();

randomList = [10,100,2910,-29,192,-19]

processedList = [x for x in randomList if x > 0 and x <= 100]
print(f"filtered and processed list: {processedList}")

makeSpace()


data = {
        "Size": [1500,2000,2500,None, 1000],
        "Bedrooms": [3,4,5,6,7],
        "Price": [300000, 400000,1000000, 500000, 600000],
        "Bathrooms": [1,2,3,4,5]
        }

df = pd.DataFrame(data)

print(f"initial datafram:\n {df}")
print (f"\nchecking for null value: \n {df.isnull().sum()}")
print (f"\nCleaned frame: \n {df.dropna()}")
print (f"\nSlice column 2 and all rows: \n{df.iloc[:, 2:3]}")


makeSpace()

list = [124.12,5.12351,1235.123,1256.63, 194.291,501925.195,19503.195,3109.591]
npList = np.array(list)

print(npList)
print(npList.shape)
print(npList.ndim)

npReshape = npList.reshape(-1,2)
print(npReshape)
print(npReshape.shape)
print(npReshape.ndim)
