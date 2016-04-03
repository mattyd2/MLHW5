import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score


def find_weighted_training_error(training_prediction, y, instance_weights):
    '''Finds the weighted training error for the predition of a given weak classifier'''

    err = 0.

    for i, w in enumerate(instance_weights):
        if training_prediction[i] != y[i]:
            err += w

    return (err * 1. / np.sum(instance_weights))


def compute_error(predicted, actual):
    '''Computes the prediction error'''

    pred = predicted.copy()

    pred[pred < 0] = -1
    pred[pred > 0] = 1

    error = np.sum(pred != actual)

    return (error * 1. / np.shape(predicted)[0])


def main():
    # Load data
    train = np.loadtxt('data/banana_train.csv', delimiter = ',')
    test = np.loadtxt('data/banana_test.csv', delimiter = ',')

    X = train[:, [1, 2]]
    y = train[:, 0]

    test_X = test[:, [1, 2]]
    test_y = test[:, 0]

    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    idx = np.arange(test_X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    test_X = test_X[idx]
    test_y = test_y[idx]

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    mean = test_X.mean(axis=0)
    std = test_X.std(axis=0)
    test_X = (test_X - mean) / std

    # Initializing equal weights to all instances
    instance_weights = np.array([1. / np.shape(X)[0]] * np.shape(X)[0])
    # print instance_weights

    # Error logs
    training_error = []
    testing_error = []

    training_prediction = np.array([0.] * np.shape(X)[0])
    test_prediction = np.array([0.] * np.shape(test_X)[0])

    for class_id in range(1, 11):

        # Fitting a decision tree and computing the model error and corresponding alpha
        clf = DecisionTreeClassifier(max_depth = 3).fit(X, y, sample_weight = instance_weights)
        # predict the values
        print X.shape
        train_prediction = clf.predict(X)
        # calculate the err_m value
        err_m = find_weighted_training_error(train_prediction, y, instance_weights)
        # calculate the alpha value
        alpha = np.log((1 - err_m) * 1. / err_m)        

        # ?????
        training_prediction += alpha * train_prediction
        # ?????
        test_prediction += alpha * clf.predict(test_X)

        # Compute errors
        train_error = compute_error(training_prediction, y)
        test_error = compute_error(test_prediction, test_y)
        # Plotting
        training_error.append(train_error)
        testing_error.append(test_error)

        # Updating the instance weights
        # print train_prediction.shape
        instance_weights[train_prediction != y] = instance_weights[train_prediction != y] * np.exp(alpha)

    # plt.plot(range(1, 11), training_error, 'r', label = "Training Error")
    # plt.plot(range(1, 11), testing_error, 'b', label = "Test Error")
    # plt.title("Training Error and Test Error with successive rounds of AdaBoost")
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()