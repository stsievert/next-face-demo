# A class designed to take the next-face-demo local, removing its reliance on
# face++ for facial recognition

import face_recognition
import pandas as pd
import numpy as np
from scipy.spatial import distance

def get_facial_landmarks(url):
    ''' gets facial landmarks for an image at the given url '''
    image = face_recognition.load_image_file(url)
    landmarks = face_recognition.face_landmarks(image)
    if len(landmarks) == 0:
        print("[FACEAPI] WARNING: No face found")
    return face_recognition.face_landmarks(image)[0]

def sort_api_data(x):
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

def normalize(face_, feature_names):
    ''' normalizes all points on the face to be based around the nose '''
    face = face_

    # find the middle point of the nose and use it for the normal point
    noseIndex = feature_names.index('nose_tip')
    normalPoints = face[noseIndex]
    normal = normalPoints[int(len(face[noseIndex])/2)] # main normal point

    # loop over each point in the face and then subtract the normal from it
    for fetIndex in [feature_names.index(f) for f in feature_names]:
        i = 0
        for pt in face[fetIndex]:
            newX = pt[0] - normal[0]
            newY = pt[1] - normal[1]
            face[fetIndex][i] = (newX, newY)
            i += 1
    # get the normalization values
    rightEyeCenterIndex = feature_names.index('right_eye')
    leftEyeCenterIndex = feature_names.index('left_eye')
    rightEye = face[rightEyeCenterIndex]
    leftEye = face[leftEyeCenterIndex]
    # subtract right from left eye
    netEye = np.subtract(rightEye, leftEye)
    # get the value of the normal
    norm = np.linalg.norm
    xNorm = norm(netEye)
    # y norm is average of normal of left and right eye
    rightNorm = norm(np.subtract(face[rightEyeCenterIndex], (0, 0)))
    leftNorm = norm(np.subtract(face[leftEyeCenterIndex], (0, 0)))
    yNorm = (rightNorm + leftNorm) / 2.0
    # apply the normalization to the x of the face values
    for fetIndex in [feature_names.index(f) for f in feature_names]:
        i = 0
        for pt in face[fetIndex]:
            newX = pt[0] / xNorm
            newY = pt[1] / yNorm
            face[fetIndex][i] = (newX, newY)
            i += 1

    # origional code ===========================================================
    # i need help making my code above more compact like this
    # # make the nose (0, 0)
    # face["x"] -= face.T["nose_tip"]["x"]
    # face["y"] -= face.T["nose_tip"]["y"]
    # # divide horizontally by distance of eyes
    # norm = np.linalg.norm
    # face["x"] /= norm(face.T["right_eye_center"] - face.T["left_eye_center"])
    #
    # # divide vertically by distance from eyes to nose
    # # works because the nose is at 0
    # vert_dist = np.mean(
    #     [norm(face.T["right_eye_center"]), norm(face.T["left_eye_center"])]
    # )
    # face["y"] /= vert_dist

    return face

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
    assert get_number_points_face(face) == 17   # we should have 17 points after this
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
