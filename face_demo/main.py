import cv2
import numpy as np
import get_words
import plot_util
import time
import base64
import io
from functools import partial
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

"""
By enabling cv2 capture, the program will attempt to capture webcam footage via
the device webcam, which is much faster than utilizing the webpage's web cam
because we do not have to communicate through javascript. However, this method
does not work over the internet (it will try to use the server's webcam, not the
users.)
"""
enable_cv2_capture = False

maxWebcamHeightCV2=240      # used for cv2
image_capture_rate = 500.0    # take an image ever n milliseconds
image_process_rate = 2000.0    # process image in python ever n seconds
capture_height = 100         # these two are used for the general javascript capture
capture_width = 200
figure_size=(1200, 800)
continue_loop = False;
loop_duration = 0.1
prev_image = ""             # previously processed image
doc = curdoc()

if enable_cv2_capture:
    take_picture_label = plot_util.make_take_picture_label()
    picture_stream_label = plot_util.make_steam_picture_label()
prime_webcam_label = plot_util.make_prime_webcam_label()
process_webcam_label = plot_util.make_process_webcam_label()
base64_label = plot_util.make_image_base_input()
title_div = plot_util.make_title_div()
description_div = plot_util.make_description_div()
github_button = plot_util.make_github_botton()

# javascript ===================================================================

# this is the javascript that takes a webcam image and saves it to the given label
# this label is then capture via python as a way to communicate from javascript to python
prime_javascript_webcam = CustomJS(args=dict(label=base64_label, process=process_webcam_label, height=capture_height, width=capture_width, time=image_capture_rate), code="""
    // canvas and context set up
    var canvas = document.createElement('CANVAS');
    document.body.appendChild(canvas);
    canvas.style.visibility = "hidden";
    const context = canvas.getContext('2d');
    context.canvas.width = width;  // set to smaller size to reduce web trafic
    context.canvas.height = height; // set to smaller size to reduce web trafic
    // video player
    var player = document.createElement('video');
    player.autoplay = true;
    player.load();
    player.controls = false;
    player.style.visibility = "hidden";
    //document.body.appendChild(player);    // hidding this fixes a bug where page would extend every time a picture is taken
    // buttons
    const captureButton = document.getElementById('capture');
    const saveButton = document.createElement("BUTTON");
    document.body.appendChild(saveButton);
    let image;

    // clear storage on refresh
    window.onbeforeunload = function(event) {
        sessionStorage.clear();
    };

    // if not there, add a false loop variable
    if (sessionStorage.getItem('loop') === null) {
        sessionStorage.setItem('loop', 'true');
    } else if (sessionStorage.getItem('loop') === 'true') {
        sessionStorage.setItem('loop', 'false');
    } else {
        sessionStorage.setItem('loop', 'true');
    }

    saveButton.addEventListener('click', () => {
        console.log("save button clicked, SHOULD be saving image");
        context.drawImage(player, 0, 0, width, height);
        image = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");
    });

    if (sessionStorage.getItem('loop') === 'true') {
        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            player.srcObject = stream;
            var interval = setInterval(function() {
                setTimeout(saveImage(), 2000);
            }, time);
        });
    }

    function saveImage() {
        saveButton.click();
        label.value = image;
        console.log("Capturing Image");
        // use this to see if the loop should continue
        var cnt = sessionStorage.getItem('loop');
        if (cnt == 'false') {
            clearInterval(interval);    // chrome only saves if this code runs
        }
    }
""")
# used with open in github button
open_github_javascript = CustomJS(code="window.open('https://github.com/stsievert/next-face-demo','_self');")

# general set up ===============================================================


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

def process_base64_image():
    '''
    processes javascript image in the base64 label
    '''
    print("[MYAPP] Capturing image via javascript webcam")

    global prev_image
    fullBase64 = base64_label.value
    trimmedBase64 = fullBase64

    # depending on how the base64 string was aqquired, we need to trime off the
    # prefix
    if trimmedBase64[:5] == "data:":
        trimmedBase64 = fullBase64[len("data:image/octet-stream;base64,"):]
    else:
        trimmedBase64 = trimmedBase64[len("b'"):]
    if trimmedBase64 != prev_image:
        prev_image = trimmedBase64
        print("[MYAPP] Setting Base64 Image of Length: " + str(len(trimmedBase64)))
        imgdata = base64.b64decode(trimmedBase64)
        image = Image.open(io.BytesIO(imgdata))
        numpImage = np.asarray(image)
        numpImage = cv2.cvtColor(numpImage, cv2.COLOR_BGR2RGB)
        callback(numpImage)
    else:
        print("[MYAPP] Image not changed, not further work being done")

def prime_webcam_clicked():
    """ primes a the webcam python call back, change button """
    if prime_webcam_label.button_type == "danger":
        prime_webcam_label.label = "Start"
        prime_webcam_label.button_type = "success"
        process_webcam_label.disabled = True
    else:
        prime_webcam_label.label = "Stop Webcam"
        prime_webcam_label.button_type = "danger"
        process_webcam_label.disabled = False


def server_webcam_callback(camera=None, delete_camera=True):
    '''
    grabs and modifys and image from the webcame to be used by the call back
    function. Potential for furthur speed increase if we can keep image in
    memory the entire time rather than reading it from a file
    - parameter camera: camera object to use
    - parameter delete_camera: if you should delete camera after use
    '''

    print("[MYAPP] Capturing image via webcam")

    camera = camera

    if camera == None:
        camera = cv2.VideoCapture(0)

    return_value, image = camera.read()

    if delete_camera == True:
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
    CV2
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
        # start loop back up
        picture_stream_callback()
    else:
        take_picture_label.disabled = False
        picture_stream_label.label = "Start Image Stream"
        picture_stream_label.button_type = "primary"

def picture_stream_callback():
    """ CV2 - Called once every time interval to continually update plot """
    camera = cv2.VideoCapture(0)
    while (continue_loop == True):
        server_webcam_callback(camera, False)
        time.sleep(loop_duration)


def setup():
    """
    General set up for the bokeh application
    """
    # put the button and plot in a layout and add to the document
    if enable_cv2_capture:
        take_picture_label.on_click(server_webcam_callback)
        picture_stream_label.on_click(toggle_picture_stream)
    github_button.js_on_event(events.ButtonClick, open_github_javascript)
    prime_webcam_label.js_on_event(events.ButtonClick, prime_javascript_webcam)
    prime_webcam_label.on_click(prime_webcam_clicked)
    process_webcam_label.on_click(process_base64_image)
    process_webcam_label.disabled = True
    if enable_cv2_capture:
        curdoc().add_root(column(widgetbox(take_picture_label), picture_stream_label, prime_webcam_label, process_webcam_label, base64_label, plot))
    else:
        curdoc().add_root(column(column(title_div, description_div), row(prime_webcam_label, process_webcam_label, github_button), plot))

setup()
#thread = Thread(target=begin_webcam_process_cycle)
#thread.start()
#test_callback()
