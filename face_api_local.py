# A class designed to take the next-face-demo local, removing its reliance on
# face++ for facial recognition

import face_recognition
import pandas as pd
import numpy as np
from scipy.spatial import distance
import numpy.linalg as LA

def get_facial_landmarks(url):
    ''' gets facial landmarks for an image at the given url '''
    image = face_recognition.load_image_file(url)
    landmarks = face_recognition.face_landmarks(image)
    if len(landmarks) == 0:
        print("[FACEAPI] WARNING: No face found")
    return face_recognition.face_landmarks(image)[0]

def sort_api_data(x: dict[list[tuple]]) -> list[list[tuple]]:
    ''' sorts API data according to the provided keys '''
    import ipdb; ipdb.set_trace()
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
def normalize(face_: list[list[tuple]], feature_names: list[str]) -> list[tuple]:
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

def reduce_data(face):
    ''' we have around 60 points but we only want 17 so this function reduces it '''
    # take one face, return the normalized data rather than directly modifying a dataframe
    # we have 60 data points but we only want 17 so take about 1/3 of the points
    range = 0.25
    while get_number_points_face(face) > 17:
        for x in face:
            if get_number_points_face(face) <= 17:
                break;
            for y in x:
                rand = np.random.uniform(0, 1)
                if rand < range:
                    x.remove(y)
                    break
    assert get_number_points_face(face) == 17
    return face

def get_number_points_face(face):
    ''' returns the number of points on the face '''
    i = 0
    for x in face:
        for y in x:
            i += 1
    return i

def convert_double_array_to_single(double):
    ''' Converts a two dimensional array to a one dimensional array '''
    single = []
    for x in double:
        for y in x:
            single.append(y)
    return single;


# run all faces through this and get that data
# use that to predict enveddubg cordinates
def distances(url):
    ''' gets distances of a face from a given url '''
    face = get_facial_landmarks(url)    # get landmarks
    face, names = sort_api_data(face)   # sort given deata
    face = normalize(face, names)       # normalize all facial data
    face = reduce_data(face)            # reduce from 60 points to 17
    face = convert_double_array_to_single(face) # transforms our array of arrays into a singular array of points
    distances = distance.pdist(face)    # distances between each point
    return distances
