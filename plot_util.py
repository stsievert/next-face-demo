# this class creates objects for the bokeh app
import numpy as np
import cv2
from bokeh.models import Label, Button

def make_image_label():
    # makes the label that will be displayed on top of the user image
    return Label(
            x = 0,
            y = 0,
            text = "Take Picture",
            border_line_color = "red",
            border_line_alpha = 0.5,
            background_fill_color = "white",
            background_fill_alpha = 0.8)

def make_take_picture_label():
    # makes the label that displays click to continue
    return Button(label="Take Picture via Webcam", button_type="success")

def process_image(img, imageHeight):
    """
    Processes image for use in myapp.py
    - parameter img: BGR ndarray of image
    - parameter imageHeight: height we want the image to be
    - returns: rgb and rgba representations of the image
    """
    # Image coming in is bigger than desired so reduce it to the inputed image height
    img = reduce_webcam_image(img, imageHeight)
    # RGB color channels are used for the face_recognition library
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # RGBA color channels are used for the bokeh library, for some reason bokeh
    # plots these images upside down, so we flip it here
    imgRGBA = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    imgRGBA = np.flipud(imgRGBA);
    return imgRGB, imgRGBA

def reduce_webcam_image(image, maxWebcamHeight):
    # reduce webcam to given size
    r = maxWebcamHeight / image.shape[1] # ratio of image
    dim = (maxWebcamHeight, int(image.shape[0] * r)) # new dimension
    imgSmall = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) # resize
    return imgSmall
