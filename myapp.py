# Imports ======================================================================
import base64
import io
import os
import sys
import traceback
import cv2  # used for webcam access
import imageio
import numpy as np
import get_words
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, CustomJS, Label, Text
from bokeh.palettes import RdYlBu3
from bokeh.plotting import curdoc, figure
from skimage.transform import resize as imresize
from show_plot import generate_initial_plot, predict, read_image

# Interface setup ==============================================================

# make webcam image WIDTH
maxWebcamHeight = 120

# Set up size of figure
FIGURE_DIM = (1200, 800)

# create a plot and style its properties
p = generate_initial_plot(test=True, n_imgs=50, img_width=0.3, dim=FIGURE_DIM)

# img = run applescript to take image
img = read_image("imgs/wanted.png")
img = img[::-1, :]

# take img and coords and update plot
y = [0, 0]

width = 0.30

WIDTH = 1.3 * width * 0.92
e = p.image_rgba(image=[img], x=[y[0]], y=[y[1]], dw=[WIDTH], dh=[WIDTH])
w = Label(
    x=0,
    y=0,
    text="Use button in upper left to upload",
    border_line_color="red",
    border_line_alpha=0.5,
    background_fill_color="white",
    background_fill_alpha=0.8,
)
# word_text = w.data_source
p.add_layout(w)
# ds_words = w.data_source
ds = e.data_source

# add a text renderer to our plot (no data yet)
r = p.text(
    x=[],
    y=[],
    text=[],
    text_color=[],
    text_font_size="20pt",
    text_baseline="middle",
    text_align="center",
)

# Funcs ========================================================================

def callback(f):
    ''' Takes in an image file and then process it to be placed onto the map '''

    w.text = "Thinking..."
    with f:
        img = imageio.imread(f.read())
        if img.shape[-1] == 3:
            shape = list(img.shape)
            shape[-1] += 1
            rgba = np.zeros(shape, dtype=img.dtype)
            rgba[..., :3] = img
            rgba[..., 3] = 255
            img = rgba
    img = img[::-1, :]
    print("IMG.SHAPE =", img.shape)
    #  imresize(img, out_shape, preserve_range=True)
    if min(img.shape[:2]) > 512:
        scale = 512 / min(img.shape[:2])
        _w, _h, _c = img.shape
        out_shape = (int(_w * scale), int(_h * scale), _c)
        img = imresize(img, out_shape, preserve_range=True).astype("uint8")
    print("IMG.SHAPE =", img.shape)
    with io.BytesIO() as f:
        imageio.imwrite(f, img, format="png")
        f.seek(0)
    try:
        y = predict('./webcam.png', verbose=True)
    except:
        err = sys.exc_info()[0]
        print("Error embedding face")
        print("**** EXCEPTION! show_plot.py#L95, error = \n{}".format(err))
        print(traceback.format_exc())
        crop_image = False
        y = np.random.randn(2)
        y /= np.linalg.norm(y) * 2

    aspect_ratio = img.shape[0] / img.shape[1]
    e.data_source.data.update(
        x=[y[0]], y=[y[1]], image=[img]  # , dw=[WIDTH], dh=[WIDTH * aspect_ratio]
    )
    emotions = get_words.find_words(y)
    print("Predicted emotions:", emotions)
    # ds_words.data.update(x=y[0], y=y[1], text=", ".join(words))
    w.x = y[0]
    w.y = y[1]
    w.text = ", ".join(emotions)

def webcam_callback():
    ''' grabs and modifys and image from the webcame to be used by the call back function '''
    ''' potential for furthur speed increase if we can keep image in memory the entire time rather than reading it from a file '''
    print("Capturing image via webcam...")
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    del(camera)

    print("Optimizing webcame image...")
    r = maxWebcamHeight / image.shape[1] # ratio of image
    dim = (maxWebcamHeight, int(image.shape[0] * r)) # new dimension
    imgSmall = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) # resize
    cv2.imwrite('webcam.png', imgSmall)

    print("Processing image...")
    with open("webcam.png", "rb") as f:
        print("Processing image....")
        callback(f)

def test_callback():
    ''' call back method used purely for testing purposes so we can test without having to lucnh the entire application'''
    with open("webcam.png", "rb") as f:
        print("Processing image....")
        callback(f)

# Trigger button ===============================================================

# put the button and plot in a layout and add to the document
demo = Button(label="Take Picture via Webcam", button_type="success")
demo.on_click(webcam_callback)
button = demo
curdoc().add_root(column(button, p))
