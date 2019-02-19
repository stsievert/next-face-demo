import cv2
import numpy as np
import get_words
import plot_util
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, CustomJS, Label, Text
from bokeh.palettes import RdYlBu3
from bokeh.plotting import curdoc, figure
from show_plot import generate_initial_plot, read_image
from face_api_local import FaceNotFoundException, predict

# parameters ===================================================================

maxWebcamHeight=240
figure_size=(1200, 800)
continue_loop = False;
take_picture_label = plot_util.make_take_picture_label()
picture_stream_label = plot_util.make_steam_picture_label()

# general set up ===============================================================


# create the plot in which we will act on
plot = generate_initial_plot(test=True, n_imgs=50, img_width=0.3,
                                                                dim=figure_size)

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
    ''' Takes in an image file and then process it to be placed onto the map '''

    img_label.text = "Thinking..."

    print("[MYAPP] Processing image")
    img_rgb, img_rgba = plot_util.process_image(img_data, maxWebcamHeight)

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

def webcam_callback():
    ''' grabs and modifys and image from the webcame to be used by the call back
    function. Potential for furthur speed increase if we can keep image in
    memory the entire time rather than reading it from a file '''

    print("[MYAPP] Capturing image via webcam")
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    del(camera)

    callback(image)

def test_callback():
    """
    if we call this in this script, we can run myapp.py on webcam.png without
    launching the entire bokeh program

    If you want to have the program automatically predict a given face on launch
    uncomment the last line in this file calling this function and then change
    the file named 'test_callback.png' to your image of choice
    """
    im = cv2.imread("test_callback.png")
    print("[MYAPP][TEST] Processing image....")
    callback(im)

def toggle_picture_stream():
    """
    Toggles if the program is continually looping images through
    """
    global continue_loop
    continue_loop = not continue_loop

    if continue_loop == True:
        # disable individual picture button so both are not running callbacks
        # at once
        take_picture_label.disabled = True
        picture_stream_label.label = "End Image Stream"
        picture_stream_label.button_type = "danger"
    else:
        take_picture_label.disabled = False
        picture_stream_label.label = "Start Image Stream"
        picture_stream_label.button_type = "primary"


def setup():
    """
    General set up for the bokeh application
    """
    # put the button and plot in a layout and add to the document
    take_picture_label.on_click(webcam_callback)
    picture_stream_label.on_click(toggle_picture_stream)
    curdoc().add_root(column(take_picture_label, picture_stream_label, plot))

setup()
#test_callback()
