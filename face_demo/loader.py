import pandas as pd

try:
    get_ipython()
    import face_demo.face_api_local as face_api
except:
    import face_api_local as face_api


def load_face_data(DIR="./train-files/", FACE_DIR="./", rad=False):
    """
    loads face data into memory and returns the data and results

    note: not used in main application, only used to help choose model
    """
    y1 = pd.read_csv(DIR + "embedding.csv", index_col="Item")

    # process all of the data
    data = []
    results = []
    for face in y1.index:
        data.append(face_api.distances(FACE_DIR + "faces/" + face + ".png"))
    for v in y1.values:
        if rad == True:
            x_calc = lambda x, y: x / (x ** 2 + y ** 2) ** 0.5
            y_calc = lambda x, y: y / (x ** 2 + y ** 2) ** 0.5
            x = v[0]
            y = v[1]
            results.append([x_calc(x, y), y_calc(x, y)])
        else:
            results.append(v)

    return data, results
