import face_api_local as face_api
from joblib import load

def predict(url, verbose=False):
    """ predicts location for a face """

    model = load("face_model.joblib")

    if verbose:
        print("[PLOT] Entering show_plot.predict")
        print("[PLOT] Finding face feature vectors...")

    x = face_api.distances(url)

    if verbose:
        print("[PLOT] Model predicting...")
    y = model.predict(x.reshape(1, -1))
    if np.linalg.norm(y) > 1:
        y /= np.linalg.norm(y)
    # y = np.clip(y, -1, 1)
    if verbose:
        print("[PLOT] Exiting show_plot.predict")
    return y.flat[:]
