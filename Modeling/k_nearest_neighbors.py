#%%
import numpy as np 
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_val_score
import pandas as pd

df = pd.read_csv('Modeling/data/breast_cancer.csv')

# replaces all ? in data set with -99999 which
# is done because most data models know that this 
# value is an outlier and disregards it
df.replace('?', -99999, inplace=True)

# drop this colum because it is not needed 
# in determianing breast cancer, not useful 
# for the other data values
df.drop(['id'], 1, inplace=True)

# %%
