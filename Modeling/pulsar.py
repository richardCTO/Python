#%%
import pandas as pd 

df = pd.read_csv('Modeling/pulsar_stars.csv')
df.head()

# %%
# Correlation heatmap
import seaborn as sns

sns.heatmap(df.corr())

# %%
import numpy as np
from sklearn.model_selection import train_test_split
X = np.array(df.drop("target_class", axis = 1))
Y = np.array(df["target_class"])

# 20% training and 80% test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2
                                                    , random_state = 42)

# %%
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

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

# %%