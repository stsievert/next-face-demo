import pandas as pd
import face_api_local as face_api

def load_face_data():
    """ loads face data into memory and returns the data and results """

    DIR = "./train-files/"
    y1 = pd.read_csv(DIR + "embedding.csv", index_col="Item")

    # process all of the data
    data = []
    results = []
    for face in y1.index:
        data.append(face_api.distances("faces/" + face + ".png"))
    for v in y1.values:
        results.append(v)

    return data, results
