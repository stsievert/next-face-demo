import numpy as np
import train
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut

## helper funcs

# tests the model we generate
data = train.testData
results = train.testDataResults
distances = train.distances
angles = train.angles


# Set up the first plot (location changes)
plt.tight_layout()
plt.subplot(2,2,1)

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
