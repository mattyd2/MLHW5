import numpy as np
import pandas as pd
import os
import decisiontree as dt


def main():
    classifiers = []

    # Question 2
    X, y = dt.prepdata('train')
    yNegOneOne = np.copy(y)
    np.place(y, y == -1, 0)
    # for depth in range(1, 11):
    #     decisiontreeclassifier = dt.decisiontree(depth, X, y)
    #     classifiers.append(decisiontreeclassifier)
    # dt.accuracyplot(classifiers, list(range(1, 11)))

    # Question 3
    initialWeights = np.ones(yNegOneOne.shape[0])/yNegOneOne.shape[0]
    dt.decisiontreeAdaBoost(X, yNegOneOne, initialWeights)
    # dt.accuracyplot(classifiers, list(range(1, 11)))

if __name__ == "__main__":
    main()
