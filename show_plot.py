import os
import random
import sys
import time
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from bokeh.client import push_session
from bokeh.driving import cosine
from bokeh.models import Label
from bokeh.plotting import curdoc, figure, output_file, show
from imageio import imread
from scipy.misc import imresize
import face_api_local as face_api
import get_words
import train_util


def read_image(filename = "faces/01F_DI_O.png", percent_scale = 0.5, center = None, width = None, height = None):
    rgba = imread(filename).astype("uint8")
    if center and width and height:
        c = (int(center[0] * rgba.shape[0] / 100), int(center[1] * rgba.shape[1] / 100))
        w = int(4 * width / 100 * rgba.shape[1])
        h = int(1.5 * height / 100 * rgba.shape[0])

        horiz = [c[0] - w // 2, c[0] + w // 2]
        vert = [c[1] - h // 2, c[1] + h // 2]
        horiz = np.clip(horiz, 0, rgba.shape[0])
        vert = np.clip(vert, 0, rgba.shape[1])

        print("[PLOT] --------------")
        print("[PLOT] " + rgba.shape)
        # rgba = rgba[horiz[0]:horiz[1], vert[0]:vert[1]]
        print("[PLOT] " + rgba.shape)
    else:
        rgba = imresize(rgba, percent_scale)

    if rgba.shape[-1] == 3:
        a = 255 * np.ones(rgba[..., 0].shape, dtype="uint8")
        out = np.zeros(rgba[..., 0].shape + (4,), dtype="uint8")
        out[..., :3] = rgba
        out[..., 3] = a
        rgba = out

    img = np.empty(rgba[:, :, 0].shape, dtype="uint32")
    view = img.view(dtype=np.uint8).reshape((rgba.shape))
    view[..., :4] = rgba[..., :4]
    return img


def generate_initial_plot(test=False, n_imgs=-1, img_width=0.5, dim=None):
    #  output_file("embedding.html")
    e = pd.read_csv("train-files/embedding.csv", index_col=0)
    e = e.reindex(np.random.permutation(e.index))
    x_lim = (-1.1, 1.3)
    y_lim = (-1.2, 1.3)

    d = []
    for filename in e.index:
        person, emotion, mouth = filename.split("_")
        coords = [e.T[filename][k] for k in ["x", "y"]]
        d += [
            {
                "person": person,
                "emotion": emotion,
                "mouth": mouth,
                "filename": filename,
                "coords": coords,
            }
        ]
    df = pd.DataFrame(d)
    d = dict(df.T)

    if dim is None:
        height = 600
        dim = (int(height * 1.6), height)
    width, height = dim
    p = figure(plot_width=width, plot_height=height, x_range=x_lim, y_range=y_lim)
    emotions = {"happy": (-1, -1), "calm": (-1, 1), "sad": (0.25, 1.0), "rage": (1, -1), "anger": (1,0.5)}
    for emotion, (x, y) in emotions.items():
        w = Label(
            x=x,
            y=y,
            text=emotion,
            text_color="red",
            text_font_size="22pt",
            # border_line_color='black', border_line_alpha=0.5,
            background_fill_color="white",
            background_fill_alpha=0.8,
        )
        p.add_layout(w)
    # locs = pd.read_csv('training_feature_locs.csv')
    for i, filename in enumerate(e.index[:n_imgs]):
        if any([face in filename for face in ["40M_AN_O", "20M_FE_O"]]):
            continue
        loc = e.loc[filename, :]
        img = read_image(filename="faces/" + filename + ".png")
        img = img[::-1, :]  # upside down because jpeg
        p.image_rgba(
            image=[img],
            x=[loc[0]],
            y=[loc[1]],
            dw=[0.62 * img_width],
            dh=[1.2 * img_width],
        )

    return p


def update_plot(img_name="webcam.png"):
    print("[PLOT] Press {enter, space} to read webcam.png")
    #  getch = Getch()
    #  key = ord(getch())  # input()
    if key in {13, 32}:
        #  print("Taking picture...")
        #  os.system("automator take_webcam_photo.workflow")
        #  print("Picture taken")
        _update_plot()


def _update_plot():
    global img_name
    crop_image = True
    img = read_image(img_name)
    img = img[::-1, :]
    print("[PLOT] Uploading image...")
    try:
        y = predict(img_name, verbose=True)
    except FaceNotFoundException:
        err = sys.exc_info()[0]
        print("[PLOT] Error embedding face")
        print("[PLOT] **** EXCEPTION! show_plot.py#L95, error = \n{}".format(err))
        print("[PLOT] " + traceback.format_exc())
        crop_image = False
        y = np.random.randn(2)
        y /= np.linalg.norm(y) * 2
    print("[PLOT] Finding the words...")
    words = get_words.find_words(y)
    print("[PLOT] " + words)
    # ds_words.data.update(x=y[0], y=y[1], text=", ".join(words))
    w.x = y[0]
    w.y = y[1]
    w.text = ", ".join(words)
    ds.data.update(x=[y[0]], y=[y[1]], image=[img])
    print("[PLOT] " + y, w)


if __name__ == "__main__":
    np.random.seed(42)
    webcam_img = False

    #  session = push_session(curdoc())
    #  no_show = True
    #  output_server('embedding')
    p = generate_initial_plot(test=True, n_imgs=50, width=0.3)

    if True:
        #  import pdb; pdb.set_trace()
        img_name = "webcam.png"

        # img = run applescript to take image
        img = read_image(img_name)
        img = img[::-1, :]

        # take img and coords and update plot
        y = [0, 0]

        width = 0.30
        e = p.image_rgba(
            image=[img], x=[y[0]], y=[y[1]], dw=[1.3 * width * 0.82], dh=[1.3 * width]
        )
        words = ["neutral", "calm"]
        w = Label(
            x=0,
            y=0,
            text=", ".join(words),
            border_line_color="red",
            border_line_alpha=0.5,
            background_fill_color="white",
            background_fill_alpha=0.8,
        )
        # word_text = w.data_source
        p.add_layout(w)
        # ds_words = w.data_source
        ds = e.data_source

    # Old implementation for 2016 WISciFest
    session = push_session(curdoc())
    session.show(p)
    curdoc().add_periodic_callback(update_plot, 1000)
    session.loop_until_closed()
