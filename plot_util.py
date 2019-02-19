# this class creates objects for the bokeh app
import numpy as np
import cv2
from bokeh.models import Label, Button

def make_image_label():
    # makes the label that will be displayed on top of the user image
    return Label(
            x = 0,
            y = 0,
            text = "Use button in upper left to upload",
            border_line_color = "red",
            border_line_alpha = 0.5,
            background_fill_color = "white",
            background_fill_alpha = 0.8)

def make_take_picture_label():
    # makes the label that displays click to continue
    return Button(label="Take Picture via Webcam", button_type="success")

def reduce_image(img):
    # if bigger than 512, reduce
    scale = 512 / min(img.shape[:2])
    _w, _h, _c = img.shape
    out_shape = (int(_w * scale), int(_h * scale), _c)
    img = imresize(img, out_shape, preserve_range=True).astype("uint8")
    return img

def reduce_webcam_image(image, maxWebcamHeight):
    # reduce webcam to given size
    r = maxWebcamHeight / image.shape[1] # ratio of image
    dim = (maxWebcamHeight, int(image.shape[0] * r)) # new dimension
    imgSmall = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) # resize
    return imgSmall

def change_color(img):
    # not sure what this does exactly
    shape = list(img.shape)
    shape[-1] += 1
    rgba = np.zeros(shape, dtype=img.dtype)
    rgba[..., :3] = img
    rgba[..., 3] = 255
    img = rgba
    img = img[::-1, :]
    return img
