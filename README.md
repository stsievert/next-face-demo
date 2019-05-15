
# NEXT Face Embedding Demo

This is a demo guesses the facial emotion of the user and maps it against a facial emotion map generate using NEXT.

![NEXT Demo Video](vids/next_face_demo.gif)

## Install

1. dlib. Instructions for install: https://github.com/ageitgey/face_recognition#installation-options
2. run `pip install -r requirements.txt`

This repo contains a model.joblib file that contains a trained estimator. To train one yourself, run the trainModel.ipynb notebook.

## Run the demo

1. Start the demo with `bokeh serve face_demo`.
2. Visit `http://localhost:5006/face_demo` (or wherever Bokeh directs you).

## Docker

Run the demo via docker by running the following command:
`docker run joeholt/next_face_demo:latest`

## How to train custom model
Using the iPython notebook `trainModel.ipynb` notebook, you can easily change the model to you liking. Change the line `model = ...` and redump the model to disk and the program will automatically use this new model.

If you are interested in testing your model before use, there is another notebook file named `testModel.ipynb` that makes the process of testing models easy. Run the existing data load code at the top of the note book and then run `test.train_and_print_metrics(results, data, YOUR_MODEL_HERE)` to get metrics of how your model scores.

