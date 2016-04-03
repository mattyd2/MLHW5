import numpy as np
import pandas as pd
import os
import decisiontree as dt


def main():
    classifiers = []

    # Question 2
    X, y = dt.prepdata('train')
    X_test, y_test = dt.prepdata('test')
    yNegOneOne = np.copy(y)
    yNegOneOne_test = np.copy(y_test)
    np.place(y, y == -1, 0)
    # for depth in range(1, 11):
    #     decisiontreeclassifier = dt.decisiontree(depth, X, y)
    #     classifiers.append(decisiontreeclassifier)
    # dt.accuracyplot(classifiers, list(range(1, 11)), X, y, X_test, y_test)

    # Question 3
    dt.decisiontreeAdaBoost(X, yNegOneOne, X_test, yNegOneOne_test)

if __name__ == "__main__":
    main()
