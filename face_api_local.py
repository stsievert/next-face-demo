# A class designed to take the next-face-demo local, removing its reliance on
# face++ for facial recognition

import face_recognition
import pandas as pd
import numpy as np
from scipy.spatial import distance

def get_facial_landmarks(url):
    ''' gets facial landmarks for an image at the given url '''
    image = face_recognition.load_image_file(url)
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

# Need help getting normalizing to work, not sure what exactly is wrong here
# gives me an error along the lines of "KeyError: 'x'"
def normalize(face_, feature_names):
    ''' normalizes all points on the face to be based around the nose '''

    face = face_
    face = pd.DataFrame(face, index=feature_names)

    # make the nose (0, 0)
    face["x"] -= face.T["nose_tip"]["x"]
    face["y"] -= face.T["nose_tip"]["y"]
    # divide horizontally by distance of eyes
    norm = np.linalg.norm
    face["x"] /= norm(face.T["right_eye_center"] - face.T["left_eye_center"])

    # divide vertically by distance from eyes to nose
    # works because the nose is at 0
    vert_dist = np.mean(
        [norm(face.T["right_eye_center"]), norm(face.T["left_eye_center"])]
    )
    face["y"] /= vert_dist
    return face.values

def distances(url):
    ''' gets distances of all faces from a given url '''
    face = get_facial_landmarks(url)
    face, names = sort_api_data(face)

    #face = normalize(face, names)  # see above for my confusion on this

    # wants a m by n array where n is two values (x,y) and m is number of points
    # the array i got above, however, isnt even close to that, its 9 by N where
    # n is the number of points under that specific point of the face
    # bellow I will convert them into a Mx2 array and see if that workds
    newFace = []
    for x in face:
        for y in x:
            newFace.append(y)
    face_dist = distance.pdist(newFace)    # find distances
    return face_dist
