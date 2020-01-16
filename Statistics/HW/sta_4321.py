#%%
import pandas as pd 
import stemgraphic

x = [
    2.9, 0.6, 13.5, 17.1, 2.8, 3.8, 16.0, 2.1, 6.4, 17.2,
7.9, 0.5, 13.7, 11.5, 2.9, 3.6, 6.1, 8.8, 2.2, 9.4,
15.9, 8.8, 9.8, 11.5, 12.3, 3.7, 8.9, 13.0, 7.9, 11.7,
6.2, 6.9, 12.8, 13.7, 2.7, 3.5, 8.3, 15.9, 5.1, 6.0
]

y = pd.Series(x)

fig, ax = stemgraphic.stem_graphic(y)

# %%
# frequency histogram

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

num_bins = 5
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
plt.show()


# %%
import statistics

x1 = [7.625, 7.500, 6.625, 7.625, 6.625, 6.875, 7.375, 5.375, 7.500]

statistics.stdev(x1)