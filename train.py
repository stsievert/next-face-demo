import json
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split

import face_api_local as face_api

DIR = "./train-files/"
y = pd.read_csv(DIR + "embedding.csv", index_col="Item")

# d is the array of faces distances given from the faces api
print("Normalizing data...")
D = {
    face: face_api.distances("faces/" + face + ".png")
    for face in y.index
    if face not in {"40M_AN_O"}
}

# y = [f(c) for x in X] where y = predict

n = len(y)
train, test = train_test_split(np.arange(n), test_size=0.1) # returns arrays of indexes that should be tested
model = KernelRidge(alpha=1 / n)

predictTestData = []
goodTestData  = []
countedPoints = 168 # for some reason the number of features flucuates between 169 and 170, different for predict and actual data, so i cap it here
# need a better way to do this...
# get train D data
i = 0
for key, value in D.items():
    if i in train and len(predictTestData) <= countedPoints:
        predictTestData.append(value)
    i += 1
# get train Y data
i = 0
for x in y.values:
    if i in train and len(goodTestData) <= countedPoints: # for some reason there is one more y than d value so we just cap it here
        goodTestData.append(x)
    i += 1
assert len(goodTestData) == len(predictTestData)
model.fit(predictTestData, goodTestData)

np.savez(DIR + "features_and_embedding.npyz", y=y, D=D)
