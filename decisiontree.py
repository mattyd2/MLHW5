import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def decisiontree(depth):
    # Parameters
    n_classes = 2
    plot_colors = "bry"
    plot_step = 0.02

    X, y = prepdata('train')

    # Train
    clf = DecisionTreeClassifier(max_depth=depth).fit(X, y)

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
    plottype = "TreeDepth" + str(depth)
    plt.savefig(plottype)
    plt.close()
    return clf


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
        df = pd.read_csv(fileDir+"/datacopy/banana_train.csv", header=None)
    elif datatypetoload == 'test':
        df = pd.read_csv(fileDir+"/datacopy/banana_test.csv", header=None)
    dfprepped = df.replace(to_replace=-1, value=0)
    dfvalues = dfprepped.values
    return dfvalues
