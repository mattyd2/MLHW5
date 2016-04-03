import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Question 2.1
def decisiontree(depth, X, y):

    # Train
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X, y)
    plotdecisiontree(X, y, clf, depth, 'TreeDepth')
    return clf


# Question 2.1
def plotdecisiontree(X, y, clf, depth, plttype):
    # Parameters
    n_classes = 2
    plot_colors = "bry"
    plot_step = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # What does this do?
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel("xlabel")
    plt.ylabel("ylabel")
    plt.axis("tight")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)

    # plt.axis("tight")
    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend()
    plottype = plttype + str(depth)
    plt.savefig(plottype)
    plt.close()


# Question 2.2
def accuracyplot(classifiers, depths, X, y, X_test, y_test):
    testErrorRate = []
    trainErrorRate = []
    for i in range(0, 10):
        testpredicted = classifiers[i].predict(X_test)
        testErrorRate.append(1-accuracy_score(y_test, testpredicted))
        trainpredicted = classifiers[i].predict(X)
        trainErrorRate.append(1-accuracy_score(y, trainpredicted))
    plt.plot(depths, testErrorRate, label="Error Test Data")
    plt.plot(depths, trainErrorRate, label="Error Train Data")
    plt.legend()
    plottype = "Q2.1.3 Performance by Tree Depth"
    plt.title(plottype)
    plt.ylabel('Error Rate')
    plt.xlabel('Depth of Tree')
    plt.savefig(plottype)
    plt.close()


# Question 3.1
def decisiontreeAdaBoost(X, y, X_test, y_test):
    errm = 0
    tot = 1
    clfs = []
    alphaMValues = []
    initWghts = np.ones(y.shape[0])/y.shape[0]

    training_error = []
    testing_error = []

    training_prediction = np.array([0.] * np.shape(X)[0])
    test_prediction = np.array([0.] * np.shape(X_test)[0])

    for m in range(0, 10):
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y, sample_weight=initWghts)
        predictions = clf.predict(X)

        incorrectPredctMask = np.invert(np.equal(predictions, y))
        errm = initWghts[incorrectPredctMask].sum()/initWghts.sum()

        # Calculate Alpha
        alpham = np.log((1-errm)/errm)

        training_prediction += alpham * predictions
        test_prediction += alpham * clf.predict(X_test)

        train_error = compute_error(training_prediction, y)
        test_error = compute_error(test_prediction, y_test)

        training_error.append(train_error)
        testing_error.append(test_error)

        # Calculate Alpha
        newWeight = np.exp(alpham)
        initWghts[predictions != y] = initWghts[predictions != y] * newWeight

    plt.plot(range(1, 11), training_error, 'r', label="Training Error")
    plt.plot(range(1, 11), testing_error, 'b', label="Test Error")
    plt.title("Q2.1.3 – Performance by Tree Depth.png")
    plt.legend()
    plt.savefig("Q2.1.3 – Performance by Tree Depth.png")


# Question 3.1
def compute_error(predicted, actual):
    pred = predicted.copy()
    pred = np.sign(pred)
    error = np.sum(pred != actual)
    return (error * 1. / np.shape(predicted)[0])


def prepdata(datatypetoload):

    # load dataframe
    data = loaddata(datatypetoload)

    # Get X and y
    X = data[:, [1, 2]]
    y = data[:, 0]

    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    return X, y


def loaddata(datatypetoload):

    # Load data
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    if datatypetoload == 'train':
        data = np.loadtxt('data/banana_train.csv', delimiter=',')
    elif datatypetoload == 'test':
        data = np.loadtxt('data/banana_test.csv', delimiter=',')
    return data
