import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import os
from joblib import load
from scipy.spatial import distance

class FaceNotFoundException(Exception):
    """ Error when API can not find a face """
    pass

model = None

def load_model():
    """ loads model file and throws error if it does not exist """
    global model
    if os.path.isfile("./face_model.joblib"):
        model = load("face_model.joblib")
    else:
        raise SystemExit("Model not found. face_model.joblib file is necessary but not found. Generate face_model.joblib by running the trainModel.ipynb notebook.")

def get_facial_landmarks(img_data):
    """
    given a file system url, returns facial landmarks for image of a face at
    said url
    - parameter img_data: image data as ndarray in RGB color format
    """
    landmarks = face_recognition.face_landmarks(img_data)

    if len(landmarks) == 0:
        raise FaceNotFoundException()

    return face_recognition.face_landmarks(img_data)[0]

def sort_api_data(x):
    #def sort_api_data(x: dict[list[tuple]]) -> list[list[tuple]]:
    """ sorts API data based on keys provided by the face finder api """
    keys = [
        "chin",
        "left_eyebrow",
        "right_eyebrow",
        "nose_bridge",
        "nose_tip",
        "left_eye",
        "right_eye",
        "top_lip",
        "bottom_lip"
    ]
    y = [x[k] for k in keys]
    return y, keys

def normalize(face_, feature_names):
    # def normalize(face_: list[list[tuple]], feature_names: list[str]) -> list[tuple]:
    """
    normalizes all points on the face to take into account a different in
    distance from the webcam for each person
    """
    face = face_

    nose_points = face[feature_names.index("nose_tip")]
    nose = nose_points[len(nose_points) // 2]  # randomly choose a point as the origin

    # setting the nose to be the origin
    face_points = [(p[0] - nose[0], p[1] - nose[1]) for group in face for p in group]

    right_eye = face[feature_names.index('right_eye')]
    left_eye = face[feature_names.index('left_eye')]

    one_unit = LA.norm(np.array(right_eye) - np.array(left_eye))

    # account for distance from camera because human faces are pretty similar
    # mostly, distance between eyes has a good relation the other facial distances
    # so, squish/expand points by the distance between the eyes
    face_points = [(p[0] / one_unit, p[1] / one_unit) for p in face_points]

    return face_points

def distances(img_data):
    """ gets distances of a face from a given url """
    face = get_facial_landmarks(img_data)    # get landmarks
    face, names = sort_api_data(face)   # sort given deata
    face = normalize(face, names)       # normalize all facial data
    distances = distance.pdist(face)    # distances between each point
    return distances

def predict(img_data):
    """ predicts location for a face given a url """

    global model

    x = distances(img_data)

    if len(x) == 0:
        raise Exception("Face Not Found")

    y = model.predict(x.reshape(1, -1))

    if np.linalg.norm(y) > 1:
        y /= np.linalg.norm(y)

    return y.flat[:]
