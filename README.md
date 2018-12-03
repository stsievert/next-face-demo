
# NEXT Face Embedding Demo

This is a demo guesses the facial emotion of the user and maps it against a facial emotion map generate using NEXT.

Start the demo with `bokeh serve myapp.py`.

Note the on macOS, only the default Terminal client has access to the webcam, other clients such as iTerm2 will not work at this time.

## Completed
1. Finding of facial features is now local
2. Using webcam rather than dropbox file upload
3. Retrain model to match changes above
4. 10x speed increases

## TODOs

1. sklearn.model_selection.LeaveOneOut¶ <- implement this

1. Further increase speed by never writing image to disk
2. Continuous image capture and plot updates via webcam
3. Plot predicted values versus actual values to get an idea of the cost
4. Warning Message when face is not found
5. Countdown timer until next image is processed
