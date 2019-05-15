# NEXT Face Embedding Demo

This is a demo guesses the facial emotion of the user and maps it against a facial emotion map generate using NEXT.

![NEXT Demo Video](vids/next_face_demo.gif)

## Run the demo

<<<<<<< HEAD
To run the webserver, run
`docker run -p XXXX:5006 joeholt/next_face_demo:latest`
This will launch a webserver listening on port XXXX of your local machine.
=======
1. Start the demo with `bokeh serve face_demo`.
2. Visit `http://localhost:5006/face_demo` (or wherever Bokeh directs you).

## Docker

Run the demo via docker by running the following command:

``` shell
docker run -p 5006:5006 ./next_face_demo:latest
```

## How to train custom model
Using the IPython notebook `trainModel.ipynb` notebook, you can easily change the model to you liking. Change the line `model = ...` and redump the model to disk and the program will automatically use this new model.
>>>>>>> 07ec611c02b21f7c2aaf23d632d90326f048450b

## Development Requirements

1. Download Anaconda
2. Install dlib
3. Run `pip install requirements.txt`
