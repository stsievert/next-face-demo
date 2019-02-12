
# NEXT Face Embedding Demo

This is a demo guesses the facial emotion of the user and maps it against a facial emotion map generate using NEXT.

Start the demo with `bokeh serve myapp.py`.

Note the on macOS, only the default Terminal client has access to the webcam. Third party clients such as iTerm2 will not work.

## How to train custom model
Using the iPython notebook `trainModel.ipynb` notebook, you can easily change the model to you liking. Change the line `model = ...` and redump the model to disk and the program will automatically use this new model.

If you are interested in testing your model before use, there is another notebook file named `testModel.ipynb` that makes the process of testing models easy. Run the existing data load code at the top of the note book and then run `test.train_and_print_metrics(results, data, YOUR_MODEL_HERE)` to get metrics of how your model scores.
