try:
    get_ipython()
    import face_demo.loader
except:
    import loader

import numpy as np
from joblib import dump
from sklearn.model_selection import LeaveOneOut

# https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def _trainModelForTrainTestSplit(train, test, data, results, model):
    """
    trains the model for one set of a train test split

    note: not used in main application, only used to help choose model
    """

    trainData = []
    trainDataResults = []
    testData = []
    testDataResults = []

    i = 0
    for value in data:
        if i in train:
            trainData.append(value)
        else:
            testData.append(value)
        i += 1
    # get train Y data
    i = 0
    for x in results:
        if i in train:
            trainDataResults.append(x)
        else:
            testDataResults.append(x)
        i += 1
    assert len(trainDataResults) == len(trainData)
    model.fit(trainData, trainDataResults)
    return testData, testDataResults


def train_model(results, data, model):
    """
    Given expected results, dace data and a model this trains the model to the data

    note: not used in main application, only used to help choose model
    """

    y = results
    D = data
    model = model
    loo = LeaveOneOut()
    loo.get_n_splits(D)
    distances = []
    angles = []
    changes = []

    for train_index, test_index in loo.split(D):
        # train model after each change
        testData, testDataResults = _trainModelForTrainTestSplit(
            train_index, test_index, D, y, model
        )
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

    return model, distances, angles, changes


def dump_model_to_disk(model, name="face_model.joblib"):
    """
    saves model to disk

    note: not used in main application, only used to help choose model
    """
    dump(model, name)
