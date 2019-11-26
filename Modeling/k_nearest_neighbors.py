#%%
import pandas as pd 

df = pd.read_csv('Modeling/data/breast_cancer.csv')
df.head()
# %%
import numpy as np
from sklearn.model_selection import train_test_split
X = np.array(df.drop("class", axis = 1))
y = np.array(df["class"])