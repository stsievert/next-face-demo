import base64
import io
import os
import sys
import traceback
from random import random

import imageio
import numpy as np
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, CustomJS, Label, Text
from bokeh.palettes import RdYlBu3
from bokeh.plotting import curdoc, figure
from skimage.transform import resize as imresize

import get_words
import upload_image
from show_plot import generate_initial_plot, predict, read_image

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


def callback(f):
    #  img_name = "webcam.png"
    #  crop_image = True
    #  img = read_image(img_name)
    img = imageio.imread("imgs/calvin.png")[::-1, :]
    e.data_source.data.update(image=[img], dh=[WIDTH])
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
        print("Uploading image...", end=" ")
        url = upload_image.upload(f, "webcam.png")
        print("done")
    try:
        y = predict(url, verbose=True)
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


# add a button widget and configure with the call back
#  button = Button(label="Press Me")
#  button.on_click(callback)
file_source = ColumnDataSource({"file_contents": [], "file_name": []})


def file_callback(attr, old, new):
    print("filename:", file_source.data["file_name"])
    raw_contents = file_source.data["file_contents"][0]
    # remove the prefix that JS adds
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = io.BytesIO(file_contents)
    callback(file_io)
    #  return file_io


file_source.on_change("data", file_callback)

upload = Button(label="Upload", button_type="success")
upload.callback = CustomJS(
    args=dict(file_source=file_source),
    code="""
function read_file(filename) {
    var reader = new FileReader();
    reader.onload = load_handler;
    reader.onerror = error_handler;
    // readAsDataURL represents the file's data as a base64 encoded string
    reader.readAsDataURL(filename);
}

function load_handler(event) {
    var b64string = event.target.result;
    file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
    file_source.trigger("change");
}

function error_handler(evt) {
    if(evt.target.error.name == "NotReadableError") {
        alert("Can't read file!");
    }
}

var input = document.createElement('input');
input.setAttribute('type', 'file');
input.setAttribute('accept', 'image/*');
input.onchange = function(){
    if (window.FileReader) {
        read_file(input.files[0]);
    } else {
        alert('FileReader is not supported in this browser');
    }
}
input.click();
""",
)


def demo_callback(*args, **kwargs):
    """
    This script takes a screenshot of quicktime. It assumes

    1. the window is in the upper left at (0, 0) of the main screen
    2. that it's taking a screenshot of an iPhone camera in iOS 12

    How to launch quicktime to record screen:

    * File > New Movie Recording
    * Click on red button and select "source: iPhone"

    """
    os.system(
        """osascript -e 'tell application "Keyboard Maestro Engine" to do script "3CF8CFB6-5A27-47FF-8DFF-F6CBA7FCE841"'"""
    )
    print("\n" + "picture taken" + "\n")
    with open("webcam.png", "rb") as f:
        callback(f)


# put the button and plot in a layout and add to the document
demo = Button(label="Take picture", button_type="success")
demo.on_click(demo_callback)

if True:
    button = upload
else:
    button = demo
curdoc().add_root(column(button, p))
