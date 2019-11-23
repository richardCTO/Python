#%%
import numpy as np 
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_val_score
import pandas as pd

df = pd.read_csv('Modeling/data/breast_cancer.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)