#%%
import pandas as pd 

df = pd.read_csv('Workshop/pulsar_stars.csv')
df.head()

# %%
# Correlation heatmap
import seaborn as sns

sns.heatmap(df.corr())

# %%
from sklearn.model_selection import train_test_split
X = df.drop("target_class", axis = 1)
Y = df["target_class"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2
                                                    , random_state = 42)

# %%
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot

clf = LogisticRegression()

clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred))

# Print R^2 and adjusted R^2
yhat = clf.predict(X)
SS_Residual = sum((Y-yhat)**2)
SS_Total = sum((Y-np.mean(Y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print (" R^2", r_squared, "Adjusted R^2", adjusted_r_squared)

# Residual Plot
# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Instantiate the linear model and visualizer
visualizer = ResidualsPlot(clf)

visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
visualizer.show() 

# %%
metrics.confusion_matrix(Y_test, y_pred)

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, Y_train)
y_pred1 = knn.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred1))

# Print R^2 and adjusted R^2
yhat = knn.predict(X)
SS_Residual = sum((Y-yhat)**2)
SS_Total = sum((Y-np.mean(Y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print (" R^2", r_squared, "Adjusted R^2", adjusted_r_squared)

# %%
metrics.confusion_matrix(Y_test, y_pred1)

# %%
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, Y_train)
y_pred1 = tree_clf.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred1))

# Print R^2 and adjusted R^2
yhat = tree_clf.predict(X)
SS_Residual = sum((Y-yhat)**2)
SS_Total = sum((Y-np.mean(Y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print (" R^2", r_squared, "Adjusted R^2", adjusted_r_squared)

# %%
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, Y_train)
y_pred2 = clf.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred2))

# Print R^2 and adjusted R^2
yhat = rfc.predict(X)
SS_Residual = sum((Y-yhat)**2)
SS_Total = sum((Y-np.mean(Y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print (" R^2", r_squared, "Adjusted R^2", adjusted_r_squared)

#%%
# only for random forest
print (rfc.feature_importances_)

# %%
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='adam', alpha=0.5, 
                    hidden_layer_sizes=(5, 5), random_state=5)

mlp.fit(X_train, Y_train)
y_pred3 = mlp.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred3))

# Print R^2 and adjusted R^2
yhat = mlp.predict(X)
SS_Residual = sum((Y-yhat)**2)
SS_Total = sum((Y-np.mean(Y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print (" R^2", r_squared, "Adjusted R^2", adjusted_r_squared)
# %%
from sklearn.svm import SVC

svm = SVC(gamma='auto')
svm.fit(X_train, Y_train)
y_pred4 = svm.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred4))

# Print R^2 and adjusted R^2
yhat = svm.predict(X)
SS_Residual = sum((Y-yhat)**2)
SS_Total = sum((Y-np.mean(Y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print (" R^2", r_squared, "Adjusted R^2", adjusted_r_squared)

# Residual Plot
# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Instantiate the linear model and visualizer

visualizer = ResidualsPlot(svm)

visualizer.fit(X_train, Y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, Y_test)  # Evaluate the model on the test data
visualizer.show() 
# %%