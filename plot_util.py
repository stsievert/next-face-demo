# this class creates objects for the bokeh app
import numpy as np
import cv2
import base64
from bokeh.models import Label, Button
from bokeh.models.widgets import TextInput, Div
#from CallbackTextInput import CallbackTextInput

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
    return Button(label="[OpenCV] Take Picture via Webcam", button_type="success")

def make_steam_picture_label():
    # makes the label that displays click to continue
    return Button(label="[OpenCV] Start Image Stream", button_type="primary")

def make_prime_webcam_label():
    # makes the label that displays click to continue
    return Button(label="Start Webcam", button_type="success")

def make_process_webcam_label():
    # makes the label that displays click to continue
    return Button(label="Take Picture and Predict Emotion", button_type="primary")

def make_image_base_input():
    return TextInput(value=default_base64, title="[JavaScript] Base64 Image Representation:"); #Callback

def make_title_div():
    return Div(text="""<b style="color:black;font-size:500%;">NEXT Face Demo</b>""", width=1000, height=60)

def make_description_div():
    return Div(text="""<font style="color:gray;font-size:150%;">Begin by starting the webcam. Once the webcam is started, take a picture and your emotion will be predicted and plotted on the map!</font>""", width=1500, height=10)

def make_github_botton():
    return Button(label="View on GitHub", button_type="default")

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

# load default image into base64
with open("imgs/wanted.png", "rb") as f:
    default_base64 = str(base64.b64encode(f.read()))
