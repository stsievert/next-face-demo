import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut

def graph_metrics(distances, angles, changes):
    """
    given an array of points for distances, an array of points for angles and
    and array of tuples for changes, this functions plots each of the given
    metrics

    note: not used in application itself, used to help with choosing model
    """

    distances = np.array(distances)

    print("Average Distance: \t " + str(distances.mean()))
    print("Mean Distance: \t\t " + str(np.median(distances)))

    plt.rcParams['figure.figsize'] = [10, 10]

    # Set up the first plot (location changes)
    plt.tight_layout()
    plt.subplot(2,2,1)

    # set up first plost, actual vs predicted face value
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.grid(True, which='both')
    plt.title('expected vs actual face location')
    max_lines = 20
    for pair in changes:
        # first value in changes = expected, 2nd = prediction
        if max_lines > 0:
            x_vals = [pair[0][0],pair[1][0]]
            y_vals = [pair[0][1],pair[1][1]]
            plt.plot(x_vals, y_vals, 'ro-')
            plt.plot(pair[0][0], pair[0][1], 'og') # expected location = green
            plt.plot(pair[1][0], pair[1][1], 'ob') # predicted location = blue
        max_lines -= 1

    # set up second plot, angle histogram
    axis = plt.subplot(2,2,2)
    axis.set_ylim([0, 80])
    plt.hist(angles, bins=10)
    plt.xlabel('Angles')
    plt.ylabel('n');
    plt.title('angle difference between results/expected')

    # set up third plot, distance histogram
    axis = plt.subplot(2, 2, 3)
    axis.set_ylim([0, 50])
    plt.ylabel('n')
    plt.xlabel('Distance Real vs Actual')
    plt.title('distance between results/expected')
    plt.hist(distances, bins=10)
    plt.show()

def plot_points(points):
    """
    given an arary of tuples of points, plots them all

    note: not used in main application, only used to help choose model
    """
    x_vals = [x[0] for x in points]
    y_vals = [x[1] for x in points]
    plt.plot(x_vals, y_vals, 'or')
    plt.show()
