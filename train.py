import json
import pickle
import loader
import numpy as np
from joblib import Parallel, delayed
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import LeaveOneOut
import show_plot as sp

# https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

y = loader.results
D = loader.data

n = len(y)
model = KernelRidge(alpha=1 / n)

def trainModelForTrainTestSplit(train, test):
    trainData = []
    trainDataResults  = []
    testData = []
    testDataResults = []

    i = 0
    for value in D:
        if i in train:
            trainData.append(value)
        else:
            testData.append(value)
        i += 1
    # get train Y data
    i = 0
    for x in y:
        if i in train:
            trainDataResults.append(x)
        else:
            testDataResults.append(x)
        i += 1
    assert len(trainDataResults) == len(trainData)
    model.fit(trainData, trainDataResults)
    return testData, testDataResults

loo = LeaveOneOut()
loo.get_n_splits(D)
distances = []
angles = []
changes = []

for train_index, test_index in loo.split(D):
    # train model after each change
    testData, testDataResults = trainModelForTrainTestSplit(train_index, test_index)
    # get variance for one test item
    d = testData[0]
    y1 = model.predict(d.reshape(1, -1))
    if np.linalg.norm(y1) > 1:
        y1 /= np.linalg.norm(y1)
    # save findings
    res = testDataResults[0]
    changes.append([testDataResults[0], y1[0]])
    distances.append(np.linalg.norm(y1[0] - res))
    angles.append(min(angle_between(y1[0], res), angle_between(res, y1[0])))
