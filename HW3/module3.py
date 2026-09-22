import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

mnist = fetch_openml('mnist_784', as_frame=False)

X, y = mnist.data, mnist.target
#print(X)
#print("----------------------------------------------\n")
#print(y.shape)

# above, we are just fetching the data
#--------------------------------------------------------

def plotDigit(imageData):
    image = imageData.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

someDigit = X[0]
plotDigit(someDigit)

#plt.show()

print(y[0])

xTrain, xTest, yTrain, yTest = X[:60000], X[60000:], y[:60000], y[60000:]

yTrain5 = (yTrain == '5')
yTest5 = (yTest == '5')

sgdCLF = SGDClassifier(random_state=42)
sgdCLF.fit(xTrain, yTrain5)
print('====================================================\n')

#print(sgdCLF.predict([someDigit]))

print(cross_val_score(sgdCLF, xTrain, yTrain5, cv=3, scoring='accuracy'))
