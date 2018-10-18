import asyncio
import os
import traceback
from pprint import pprint

import numpy as np
import pandas as pd
import requests
from scipy.spatial import distance

import upload_image

KEY = os.environ.get("FACEPP_KEY", None)
SECRET = os.environ.get("FACEPP_SECRET", None)
if KEY is None or SECRET is None:
    raise ValueError(
        "Face++ secret and key are missing. Generate one at"
        "https://www.faceplusplus.com/ and export as FACEPP_KEY"
        "and FACEPP_SECRET"
    )


def face_api_response(image_url, n_features=25, repeat=2):
    if repeat < 0:
        raise Exception("Recursion limit reached in image upload")
    try:
        args = {"url": image_url}
        print("   begin POST to find faces img URL")
        r1 = requests.post(
            "https://faceplusplus-faceplusplus.p.mashape.com/detection/detect",
            data=args,
            headers={"X-Mashape-Key": KEY, "Accept": "application/json"},
        )
        print("   end POST to find faces img URL")

        if "face" not in r1.json().keys() or len(r1.json()["face"]) == 0:
            print("r1.json().keys() = ", r1.json().keys())
            print("face not found in ", image_url)
            print("error = {}".format(r1.json()["error"]))
            if r1.json()["error"] == "SERVER_TOO_BUSY":
                print("server too busy, retrying...")
                return face_api_response(image_url)
            raise Exception("Face API didn't find a face")

        face_id = r1.json()["face"][0]["face_id"]

        print("   begin POST to find face landmarks in face")
        r = requests.get(
            "https://faceplusplus-faceplusplus.p.mashape.com/detection/landmark?face_id={}&type={}p".format(
                face_id, n_features
            ),
            headers={
                "X-Mashape-Key": "pLToTGrwPZmshNf4EjL0nWPg3QQTp1ofWuAjsnKy8Qv2eAptNN",
                "Accept": "application/json",
            },
        )
        print("   end POST to find face landmarks in face")

        if "result" not in r.json().keys() or len(r.json()["result"]) == 0:
            print("\n   *** error! result not in r.keys()! for \n     ", image_url)
            return face_api_response(image_url, repeat=repeat - 1)

        return r.json()["result"][0]["landmark"]
    except:
        print("\n    Some exception! Doing recursion with repeat=", repeat)
        print(traceback.format_exc())
        return face_api_response(image_url, repeat=repeat - 1)


def sort_api_response(x):
    keys = [
        "left_eye_left_corner",
        "nose_tip",
        "right_eye_center",
        "right_eye_left_corner",
        "mouth_lower_lip_bottom",
        "mouth_right_corner",
        "mouth_left_corner",
        "left_eye_center",
        "left_eyebrow_left_corner",
        "left_eye_pupil",
        "mouth_upper_lip_bottom",
        "right_eye_pupil",
        "left_eye_right_corner",
        "mouth_lower_lip_top",
        "right_eyebrow_left_corner",
        "mouth_upper_lip_top",
        "nose_left",
        "left_eye_bottom",
        "nose_right",
        "right_eye_top",
        "left_eye_top",
        "right_eye_right_corner",
        "left_eyebrow_right_corner",
        "right_eye_bottom",
        "right_eyebrow_right_corner",
    ]

    y = [x[k] for k in keys]
    return y, keys


def normalize(face_, feature_names):
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


def facepp_response(url):
    creds = {"key": KEY, "secret": SECRET}
    r = requests.post(
        "https://api-us.faceplusplus.com/facepp/v3/detect",
        params={
            "api_key": creds["key"],
            "api_secret": creds["secret"],
            "image_url": url,
            "return_landmark": 1,
        },
    )
    assert r.status_code == 200, "Request failed"
    faces = r.json()["faces"]
    if len(faces) == 0:
        print("Response:\n")
        pprint(r.json())
        raise ValueError("No faces found in image")
    return faces[0]["landmark"]


def distances(url):
    print("Getting the faceplusplus response...", end=" ")

    # Mashape API was broken for faceplusplus on 2018-09-27
    # had to resort to raw faceplusplus API
    #  face = face_api_response(url)
    face = facepp_response(url)
    print("done")

    face, names = sort_api_response(face)
    face = normalize(face, names)
    face_dist = distance.pdist(face)
    return face_dist


if __name__ == "__main__":
    d = distances("./01F_DI_C.png")
