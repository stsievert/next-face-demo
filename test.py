import numpy as np
import train
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut

plt.rcParams['figure.figsize'] = [10, 10]

# tests the model we generate
data = train.testData
results = train.testDataResults
distances = train.distances
angles = train.angles
changes = train.changes


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
    if max_lines > 0:
        plt.plot(pair[0], pair[1], 'ro-')
    max_lines -= 1

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
