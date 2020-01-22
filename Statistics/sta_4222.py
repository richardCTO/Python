#%%
import pandas as pd 

df = pd.read_csv('/home/richard/Documents/Python/Statistics/HW/Data/ClassSurvey.csv')
df.head()

# %%
# Correlation heatmap
import seaborn as sns

sns.heatmap(df.corr())

# %%
import numpy as np

