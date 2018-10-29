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
if False:
    D = {
        face: face_api.distances("faces/" + face + ".png")
        for face in y.index
        if face not in {"40M_AN_O"}
    }

    with open(DIR + "D.pkl", "wb") as f:
        pickle.dump(D, f)
else:
    with open(DIR + "D.pkl", "rb") as f:
        D = pickle.load(f)

    y = dict(y.T)
    y = {key: value for key, value in y.items() if key in D.keys()}
    y = pd.DataFrame(y).T
    D = pd.DataFrame(D).T
    assert len(D) == len(y)
    len_before = len(D)
    cols = D.columns
    df = pd.merge(y, D, left_index=True, right_index=True)
    y = df[["x", "y"]]
    D = df[cols]
    assert len(D) == len_before

    assert all(y.index == D.index), "Make sure the features and items are ordered!"
    y = y.values
    D = D.values

n, p = len(y), D.shape[1]
train, test = train_test_split(np.arange(n), test_size=0.1)
model = KernelRidge(alpha=1 / n)
model.fit(D[train], y[train])

np.savez(DIR + "features_and_embedding.npyz", y=y, D=D)

if __name__ == "__main__":
    y_hat = model.predict(D[test])

    distance = np.linalg.norm(y_hat - y[test], axis=1)
    distance /= y.max() - y.min()
    print("median distance", np.median(distance))

    import matplotlib.pyplot as plt

    plt.figure()
    for y_h, yi in zip(y_hat, y):
        plt.plot(*y_h, "ro")
        plt.plot(*yi, "bo")
        plt.plot([yi[0], y_h[0]], [yi[1], y_h[1]], "y--")
    plt.show()
