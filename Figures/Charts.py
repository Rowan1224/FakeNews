import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

# y-axis in bold
rc('font', weight='bold')

# Values of each group
# bars1 = [81,23, 90,50, 86,32, 85,52, 85,52, 89,63]
# bars2 = [19,76, 10,50, 14,67, 15,48, 15,48, 11,36]

bars1 = [81,23, 90,50, 86,32, 85,52, 85,52, 89,63]
bars2 = [19,76, 10,50, 14,67, 15,48, 15,48, 11,36]

# Heights of bars1 + bars2
# bars = np.add(bars1, bars2).tolist()

# The position of the bars on the x-axis
r = [0,0.2, 0.6,0.8, 1.2,1.4, 1.8,2, 2.4,2.6, 3,3.2]

# Names of group and bar width
names = ['Unigram','', 'C3-Gram','', '','C3+C4+C5', '','L', '','L+E', '','AllFeatures' ]
barWidth = 0.2

# Create brown bars
plt.bar(r, bars1, color=['#ec7063', '#5dade2'], edgecolor='white', width=barWidth)
# Create green bars (middle), on top of the firs ones
plt.bar(r, bars2, bottom=bars1, color=['#feffc4', '#d5f5e3'], edgecolor='white', width=barWidth)

# Custom X axis
plt.xticks(r, names)
plt.xlabel("\nSVM & LR Comparison", fontweight='bold')
plt.ylabel("F1", fontweight='bold')

# Show graphic
plt.legend(['SVM','LR'])
plt.show()



