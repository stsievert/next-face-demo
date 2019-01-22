
# NEXT Face Embedding Demo

This is a demo guesses the facial emotion of the user and maps it against a facial emotion map generate using NEXT.

Start the demo with `bokeh serve myapp.py`.

Note the on macOS, only the default Terminal client has access to the webcam, other clients such as iTerm2 will not work at this time.

## Completed
1. Finding of facial features is now local
2. Using webcam rather than dropbox file upload
3. Retrain model to match changes above
4. 10x speed increases
5. Implement LeaveOneOut for training
6. Implementing LASSO Model
6. Testing via various metrics

## TODOs

1. Further increase speed by never writing image to disk
2. Continuous image capture and plot updates via webcam
3. Plot predicted values versus actual values to get an idea of the cost
4. Warning Message when face is not found
5. Countdown timer until next image is processed



End goal: Find a good value of alpha


DO THIS FOR SURE
6. Make sure normalization is working
6.1. Fix normalization
7. Fix weird y compresion

DONT NEED THIS ANYMORE
7. Select best features (using RFS or SelectFromModel)
7.1. When doing select best features, give it ALL faces with ALL features and all results and run it through
7.2. May not need feature selection for Lasso, it works on changing alpha rather than changing features

DO THIS AFTER FIXING
8. Use Lasso and choose a bunch of different alpha values, dont use LassoCV unless someting work
- check normal Lasso with lots of different a values and see if we can tune it to work nicely from there
- 8.1.1. Make a graph of alpha (x axis) and its score (y axis, get score from SciKitLearn score value feeding it the test test test)
- 8.1.2. Find bounderies of where Alpha is good / bad, see if there are, basically see whats up with alpha

NO LONGER NEED THIS, DO IF HAVE TIME
- 8. Use [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV) instead, this picks alpha for you
- 8.1. Before this, check normal Lasso with lots of different a values and see if we can tune it to work nicely from there
- 8.1.1. Make a graph of alpha (x axis) and its score (y axis, get score from SciKitLearn score value feeding it the test test test)

DO THIS AFTER LASSO
9. Use GridSearchEstimator/RandomSearchCV, this can give us an alpha, use it try to find alpha, best alpha
9.1 Use Lasso to pass as an estimator
9.2 for param grid do something like {allpha: [0.1, 1, 10, 100]]}, try a variety of different things to pass to it
9.3 cv splitter: pass it lea e one out



