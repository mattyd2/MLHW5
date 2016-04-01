import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def decisiontree(depth, X, y):

    # Train
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X, y)
    plotdecisiontree(X, y, clf, depth, 'TreeDepth')
    return clf


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
def accuracyplot(classifiers, depths):
    X, y = prepdata('test')
    treeErrorRate = []
    for i in range(0, 10):
        y_predicted = classifiers[i].predict(X)
        treeErrorRate.append(1-accuracy_score(y, y_predicted))
    plottype = "TreeDepth by Performance on Test Data"
    plt.plot(depths, treeErrorRate, label="Depth by Accuracy on Test Data")
    plt.legend()
    plt.title(plottype)
    plt.ylabel('Error Rate')
    plt.xlabel('Depth of Tree')
    plt.savefig(plottype)
    plt.close()


def decisiontreeAdaBoost(X, y, initialWeights):
    err = 0
    tot = 1
    clfs = []
    alphaMValues = []
    for m in range(0, 10):
        clf = DecisionTreeClassifier(max_depth=3)
        print initialWeights
        clf.fit(X, y, sample_weight=initialWeights)
        # Plot the tree for this iteration
        # plotdecisiontree(X, y, clf, 3, 'AdaBoost')
        predictedValueM = clf.predict(X)
        # Add the prediction to list
        clfs.append(clf)
        incorrectPredctMask = np.invert(np.equal(predictedValueM, y.ravel()))
        W = np.copy(initialWeights)
        errm = initialWeights[incorrectPredctMask].sum()/W.sum()
        alpham = np.log((1.-errm)/errm)
        # Add alpham value to list
        alphaMValues.append(alpham)
        newWeight = np.exp(alpham)
        np.putmask(initialWeights, incorrectPredctMask, np.array(newWeight))
    return accuracyplotAdaBoost(X, y, clfs, alphaMValues)


# Question 3.3
def accuracyplotAdaBoost(X, y, clfs, alphaMValues):
    treeErrorRate = []
    for m in range(0, 10):
        y_predicted = clfs[m].predict(X)
        treeErrorRate.append(1-accuracy_score(y, y_predicted))
    plottype = "AdaBoost Performance on Test Data"
    plt.plot(range(0, 10), treeErrorRate, label="Accuracy of AdaBoost on Test")
    plt.legend()
    plt.title(plottype)
    plt.ylabel('Error Rate')
    plt.xlabel('Number of Iterations (m)')
    plt.savefig(plottype)
    plt.close()


def prepdata(datatypetoload):

    # load dataframe
    dfvalues = loaddata(datatypetoload)

    # We only take the two corresponding features
    X = dfvalues[:, 1:3]
    y = dfvalues[:, 0:1]

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
        df = pd.read_csv(fileDir+"/data/banana_train.csv", header=None)
    elif datatypetoload == 'test':
        df = pd.read_csv(fileDir+"/data/banana_test.csv", header=None)
    dfvalues = df.values
    return dfvalues
