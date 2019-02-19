import base64
import io
import os
import sys
import traceback
import cv2
import imageio
import numpy as np
import get_words
import plot_util
from PIL import Image
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, CustomJS, Label, Text
from bokeh.palettes import RdYlBu3
from bokeh.plotting import curdoc, figure
from show_plot import generate_initial_plot, read_image
from face_api_local import FaceNotFoundException, predict

maxWebcamHeight = 240       # captured webcam image height
FIGURE_DIM = (1200, 800)    # Set up size of figure
plot = generate_initial_plot(test=True, n_imgs=50, img_width=0.3, dim=FIGURE_DIM)  # plot

# had to open in PIL to get a ndarray representation rather than iobuffer while also loading in RGBA color space
im = Image.open('imgs/wanted.png')
im = im.convert("RGBA")
imarray = np.array(im)
imarray = np.flipud(imarray)

# plot the image
width = 0.3588 # 1.3 * 0.30 * 0.92
e = plot.image_rgba(image=[imarray], x=[0], y=[0], dw=[width], dh=[width])

# add the image label to the plot
img_label = plot_util.make_image_label()
plot.add_layout(img_label)
ds = e.data_source

# add a text renderer to our plot (no data yet)
text_renderer = plot.text(x = [], y = [], text = [], text_color = [], text_font_size="20pt", text_baseline="middle", text_align="center")


def callback(img_data):
    ''' Takes in an image file and then process it to be placed onto the map '''

    print("[MYAPP] Processing image callback")
    img_label.text = "Thinking..."

    # what does this do?
    if min(img_data.shape[:2]) > 512:
        img_data = plot_util.reduce_image(img_data)

    try:
        print("[MYAPP] Attempting prediction")
        y = predict(img_data)
    except FaceNotFoundException as exc:
        print("[MYAPP] Face not found, making random guess")
        y = np.random.randn(2)
        y /= np.linalg.norm(y) * 2

    aspect_ratio = img_data.shape[0] / img_data.shape[1]

    img_data_rgba = cv2.cvtColor(img_data, cv2.COLOR_RGB2RGBA)
    img_data_rgba_flipped = np.flipud(img_data_rgba)
    e.data_source.data.update(x = [y[0]], y = [y[1]], image = [img_data_rgba_flipped])

    emotions = get_words.find_words(y)
    print("[MYAPP] Predicted emotions: ", emotions)
    img_label.x = y[0]
    img_label.y = y[1]
    img_label.text = ", ".join(emotions)

def webcam_callback():
    ''' grabs and modifys and image from the webcame to be used by the call back
    function. Potential for furthur speed increase if we can keep image in
    memory the entire time rather than reading it from a file '''

    print("[MYAPP] Capturing image via webcam")
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    del(camera)

    print("[MYAPP] Reducing webcame image")
    imgSmall = plot_util.reduce_webcam_image(image, maxWebcamHeight)
    imgRGB = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    callback(imgRGB)


def test_callback():
    """
    if we call this in this script, we can run myapp.py on webcam.png without
    launching the entire bokeh program
    """
    im = cv2.imread("test_callback.png", 0)     # loads in grayscale color space
                                                # tried rgb but it failed for some reason with the error "RuntimeError: Unsupported image type, must be 8bit gray or RGB image."
    # imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # now in rgb color space
    print("[MYAPP][TEST] Processing image....")
    callback(im)

def setup():
    # put the button and plot in a layout and add to the document
    take_picture_label = plot_util.make_take_picture_label()
    take_picture_label.on_click(webcam_callback)
    curdoc().add_root(column(take_picture_label, plot))
    #webcam_callback()

setup()

#test_callback()
