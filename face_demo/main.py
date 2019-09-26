import cv2
import numpy as np
import get_words
import plot_util
import time
import base64
import io
import threading
from functools import partial
from tornado.ioloop import PeriodicCallback
from PIL import Image
from bokeh import events
from bokeh.io import show
from bokeh.layouts import column, row, widgetbox
from bokeh.models import Button, ColumnDataSource, CustomJS, Label, Text
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import TextInput
from bokeh.models.tools import SaveTool
from bokeh.palettes import RdYlBu3
from bokeh.plotting import curdoc, figure
from show_plot import generate_initial_plot, read_image
from face_api_local import FaceNotFoundException, predict

# parameters ===================================================================

curdoc().title = "NEXT Face Demo"

maxWebcamHeightCV2=240      # used for cv2
image_capture_rate = 500.0    # take an image ever n milliseconds
image_process_rate = 2000.0    # process image in python ever n seconds
capture_height = 100         # these two are used for the general javascript capture
capture_width = 200
figure_size=(1200, 800)
continue_loop = False; # NOT a config variable
loop_duration = 0.5
prev_image = ""             # previously processed image
doc = curdoc()
camera = cv2.VideoCapture(0)

take_picture_label = plot_util.make_take_picture_label()
picture_stream_label = plot_util.make_steam_picture_label()
title_div = plot_util.make_title_div()
description_div = plot_util.make_description_div()

# create the plot in which we will act on
plot = generate_initial_plot(test=True, n_imgs=50, img_width=0.3,
                                                                dim=figure_size)
plot.toolbar.active_drag = None
plot.toolbar.active_scroll = None
plot.toolbar.active_tap = None
plot.toolbar.logo = None
plot.tools = [SaveTool()]

# load the default/starting image and move it to RGBA (bokeh) color format
# we must flip it because for some reason bokeh plots images upside down
im = cv2.imread('imgs/wanted.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
imarray = np.flipud(im)
e = plot.image_rgba(image=[imarray], x=[0], y=[0], dw=[0.3588], dh=[0.3588])

# add the image label to the plot
img_label = plot_util.make_image_label()
plot.add_layout(img_label)
ds = e.data_source

# add a text renderer to our plot (no data yet)
text_renderer = plot.text(x = [], y = [], text = [], text_color = [],
             text_font_size="20pt", text_baseline="middle", text_align="center")

# callbacks ====================================================================


def callback(img_data):
    ''' Takes in an image file and then process it to be placed onto the map
        takes in BGR image
    '''

    img_label.text = "Thinking..."

    print("[MYAPP] Processing image")
    img_rgb, img_rgba = plot_util.process_image(img_data, maxWebcamHeightCV2)

    print("[MYAPP] Attempting prediction")
    try:
        y = predict(img_rgb)
        print("[MYAPP] Prediction successful")
    except FaceNotFoundException as exc:
        print("[MYAPP] Face not found, making random guess")
        y = np.random.randn(2)
        y /= np.linalg.norm(y) * 2

    # update image on plot
    e.data_source.data.update(x = [y[0]], y = [y[1]], image = [img_rgba])
    # update image label
    emotions = get_words.find_words(y)
    img_label.x = y[0]
    img_label.y = y[1]
    img_label.text = ", ".join(emotions)
    print("[MYAPP] Predicted emotions: ", emotions)

def server_webcam_callback(delete_camera=True):
    '''
    CV2
    grabs and modifys and image from the webcame to be used by the call back
    function. Potential for furthur speed increase if we can keep image in
    memory the entire time rather than reading it from a file
    - parameter camera: camera object to use
    - parameter delete_camera: if you should delete camera after use
    '''

    print("[MYAPP] Capturing image via webcam")

    global camera

    if camera == None:
        camera = cv2.VideoCapture(0)

    return_value, image = camera.read()

    if delete_camera == True:
        del(camera)

    callback(image)

def toggle_picture_stream():
    """
    CV2
    Toggles if the program is continually looping images through
    """
    global continue_loop
    global take_picture_label
    global picture_stream_label
    continue_loop = not continue_loop

    if continue_loop == True:
        # disable individual picture button so both are not running callbacks
        # at once
        take_picture_label.disabled = True
        picture_stream_label.label = "[OpenCV] End Image Stream"
        picture_stream_label.button_type = "danger"
        # start loop back up
        picture_stream_callback()
    else:
        take_picture_label.disabled = False
        picture_stream_label.label = "[OpenCV] Start Image Stream"
        picture_stream_label.button_type = "primary"

def update():
    """
    Updates display with conntinuous webcam input
    """
    if continue_loop:
        server_webcam_callback(False)


def setup():
    """
    General set up for the bokeh application
    """
    # put the button and plot in a layout and add to the document
    take_picture_label.on_click(server_webcam_callback)
    picture_stream_label.on_click(toggle_picture_stream)
    curdoc().add_root(column(title_div, description_div, picture_stream_label, take_picture_label, plot))
    curdoc().add_periodic_callback(update, loop_duration)

setup()
