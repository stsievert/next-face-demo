import numpy as np
import train
import matplotlib.pyplot as plt

## helper funcs

# https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

## other

MAX_PRINT = 5

# tests the model we generate
data = train.testData
results = train.testDataResults

distances = []
angles = []

# Set up the first plot (location changes)
plt.tight_layout()
plt.subplot(2,2,1)

i = 0;
for d in data:
    y = train.model.predict(d.reshape(1, -1))
    if np.linalg.norm(y) > 1:
        y /= np.linalg.norm(y)
    # save findings
    distances.append(np.linalg.norm(y[0] - results[i]))
    angles.append(min(angle_between(y[0], results[i]), angle_between(results[i], y[0])))
    # plot
    plt.plot(y[0], results[i])

    i += 1

plt.xlabel('x')
plt.ylabel('y')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid(True, which='both')
plt.title('expected vs actual face location')

# set up second plot, angle histogram
plt.subplot(2,2,2)
plt.hist(angles, bins=10)
plt.xlabel('Angles')
plt.ylabel('n');
plt.title('angle difference between results/expected')

# set up third plot, distance histogram
plt.subplot(2, 2, 3)
plt.ylabel('n')
plt.xlabel('Distance Real vs Actual')
plt.title('distance between results/expected')
plt.hist(distances, bins=10)

plt.show()
