#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

pokemon_data = pd.read_csv("Modeling/data/pokemon/pokemon.csv")
pokemon_data.head(10)
pokemon_data.tail()

# %%
# prints out all the columns for the data set
pokemon_data.columns

# %%
