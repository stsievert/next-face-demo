import numpy as np
import pandas as pd

def find_words(y):
    df = pd.read_csv("./word-coords/words_2D_aligned_coarse.csv", index_col="word")
    df -= y
    norms = np.linalg.norm(df, axis=1)
    df["diff norm"] = norms

    i = np.argsort(norms)[:2]

    words = list(df.iloc[i].T.keys())
    return words
