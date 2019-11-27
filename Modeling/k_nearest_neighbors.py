#%%
import pandas as pd 

df = pd.read_csv('Modeling/data/breast_cancer.csv')
df.replace('?', -99999, inplace=True)
df.head()
# %%
import numpy as np
from sklearn.model_selection import train_test_split

# Remove the class column, as this could
# affect the data training 
X = np.array(df.drop("class", axis = 1))
y = np.array(df["class"])

# 20% training and 80% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# This is the classifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy", metrics.accuracy_score(y_test, y_pred))

# %%
