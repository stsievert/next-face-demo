# A class designed to take the next-face-demo local, removing its reliance on
# face++ for facial recognition

import face_recognition
import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy.linalg as LA

def get_facial_landmarks(url):
    ''' gets facial landmarks for an image at the given url '''
    image = face_recognition.load_image_file(url)
    landmarks = face_recognition.face_landmarks(image)
    if len(landmarks) == 0:
        print("[FACEAPI] WARNING: No face found")
    return face_recognition.face_landmarks(image)[0]

def sort_api_data(x):
    #def sort_api_data(x: dict[list[tuple]]) -> list[list[tuple]]:
    ''' sorts API data according to the provided keys '''
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

# test this by ploting all the points from one face
# then normalize it, and plot it again, it should be about the same then
def normalize(face_, feature_names):
    # def normalize(face_: list[list[tuple]], feature_names: list[str]) -> list[tuple]:
    """ normalizes all points on the face to be based around the nose """
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

def plot_points(points):
    """ given an arary of tuples of points, plots them all  """
    x_vals = [x[0] for x in points]
    y_vals = [x[1] for x in points]
    plt.plot(x_vals, y_vals, 'or')
    plt.show()

# run all faces through this and get that data
# use that to predict enveddubg cordinates
def distances(url):
    ''' gets distances of a face from a given url '''
    face = get_facial_landmarks(url)    # get landmarks
    face, names = sort_api_data(face)   # sort given deata
    face = normalize(face, names)       # normalize all facial data
    distances = distance.pdist(face)    # distances between each point
    return distances
