import numpy as np
import pandas as pd


def find_words(y):

    """
    Show two words that correspond with the face image taken by webcame

    Parameters
    ----------
    y : ndarray
        of length 2 (coordinates of face)

    returns
    -------
    words : List[str]
        Two words that correspond with a facial emotion
    """
    df = pd.read_csv("./word-coords/words_2D_aligned_coarse.csv", index_col="word")

    coords = [coords for coords in df.values]
    words = [word for word in df.index]
    norms = [np.linalg.norm(coord - y) for coord in coords]
    smallest_two_idx = np.argsort(norms)[:2]  # .argsort()[:2]
    words = [words[smallest_two_idx[0]], words[smallest_two_idx[1]]]
    return words


if __name__ == "__main__":
    words = find_words([0, 0])
    assert words == ["disgust", "sadness"]
