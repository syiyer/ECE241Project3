import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

training = pd.read_csv("train.csv")
testing = pd.read_csv('test.csv')
# Importing data and reading as csv file

records = training['Price'].count()
print("Amount of records - %s " % records)
# gives amount of rows in dataset (also # of records)

meanPrice = training['Price'].mean()
print("Mean price - %s " % meanPrice)
# gives mean of price column

minPrice = training['Price'].min()
print("Minimum price - %s " % minPrice)
# gives smallest value in price column

maxPrice = training['Price'].max()
print("Maximum price - %s " % maxPrice)
# gives highest value in price column

stDev = training['Price'].std()
print("Standard deviation - %s " % stDev)
# gives standard deviation of price column

col = ['Price']
plt.hist(training[col])
plt.xlabel('Price')
plt.ylabel('Occurrences')
plt.title('Histogram of Price')
plt.show()
# outputs histogram of housing prices column

cols = ['GrLivArea', 'BedroomAbvGr', 'TotalBsmtSF', 'FullBath']
sns.pairplot(training[cols], height=1.5)
plt.show()
# outputs pairwise scatterplot of 4 different features


def pred(features, weights):
    pricePred = np.dot(features, weights)
    # gives the predicted price column vector based on the features and weights
    return pricePred


def loss(correctPrice, predPrice):
    meanSquareError = np.mean((predPrice - correctPrice)**2)
    # gives mean total mean square error from predicted and exact price vectors
    return meanSquareError


def gradient(weights, correctPrice, features):
    predPrice = np.dot(features, weights)
    # calculates predicted price
    error = predPrice - correctPrice
    # calculates error between actual and predicted price
    gradient = (2/len(correctPrice)) * np.dot(np.transpose(features), error)
    # gives gradient for dataset based on values calculated and gradient formula
    return gradient


def update(gradient, alpha, weights):
    weight = weights - (alpha*gradient)
    # updates new weights based on alpha value and gradient outputted
    return weight


def trainModel(features, correctPrice, weights, alpha1, alpha2, iterations):
    mseValues1 = []
    mseValues2 = []
    # initiates empty lists to append individual MSE values from each iteration
    weights1 = weights
    weights2 = weights
    # sets weights starting value for each alpha value as the same
    for i in range(iterations):
        prediction1 = pred(features, weights1)
        prediction2 = pred(features, weights2)
        # calculates predictions vectors from pred function
        mse1 = loss(correctPrice, prediction1)
        mse2 = loss(correctPrice, prediction2)
        # calculates MSEs from loss function
        grad1 = gradient(weights1, correctPrice, features)
        grad2 = gradient(weights2, correctPrice, features)
        # calculates gradients from gradient function
        weights1 = update(grad1, alpha1, weights1)
        weights2 = update(grad2, alpha2, weights2)
        # updates new weights from update function
        print("MSE1 for %s iterations - %s" % (i, mse1))
        print("MSE2 for %s iterations - %s" % (i, mse2))
        # prints MSE values for individual iteration
        mseValues1.append(mse1)
        mseValues2.append(mse2)
        # appends new MSE values for each iteration
    plt.plot(mseValues1)
    plt.plot(mseValues2)
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("Learning Curves for different learning rates")
    plt.legend(["Learning rate - %s" % alpha1, "Learning rate - %s" % alpha2])
    plt.show()
    # outputs plots for MSE values and iterations


alpha1 = 8e-10
alpha2 = 1e-9
# generate alpha values for training
iterations = 500
# generate amount of iterations for training
features = training.iloc[:, 1:-1].values
# generates features matrix from training data
correctPrice = training['Price'].values
# generates price column vector taken from training data
weights = np.random.rand(features.shape[1])
# generates random weight vector
trainModel(features, correctPrice, weights, alpha1, alpha2, iterations)
# run training model main function with corresponding values

