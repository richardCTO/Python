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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import numpy as np
from sklearn.model_selection import train_test_split
X = np.array(df.drop("class", axis = 1))
y = np.array(df["class"])

# 20% training and 80% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
print("Accuracy", metrics.accuracy_score(y_test, y_pred1))

# %%
