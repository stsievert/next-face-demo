import pandas as pd
import face_api_local as face_api

# get data and load it into a dataframe
DIR = "./train-files/"
y1 = pd.read_csv(DIR + "embedding.csv", index_col="Item")

# process all of the data
data = []
results = []
for face in y1.index:
    data.append(face_api.distances("faces/" + face + ".png"))
for v in y1.values:
    results.append(v)

print("Data successfully loaded")
