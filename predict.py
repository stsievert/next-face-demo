
def predict(url, verbose=False):
    if verbose:
        print("[PLOT] Entering show_plot.predict")
        print("[PLOT] Finding face feature vectors...")
    x = face_api.distances(url)
    if verbose:
        print("[PLOT] Model predicting...")
    y = train.model.predict(x.reshape(1, -1))
    if np.linalg.norm(y) > 1:
        y /= np.linalg.norm(y)
    # y = np.clip(y, -1, 1)
    if verbose:
        print("[PLOT] Exiting show_plot.predict")
    return y.flat[:]
