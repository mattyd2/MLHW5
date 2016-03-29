import numpy as np
import pandas as pd
import os
from decisiontree import decisiontree, accuracyplot


def main():
    classifiers = []
    for i in range(1, 11):
        decisiontreeclassifier = decisiontree(i)
        classifiers.append(decisiontreeclassifier)
    accuracyplot(classifiers, list(range(1, 11)))

if __name__ == "__main__":
    main()
