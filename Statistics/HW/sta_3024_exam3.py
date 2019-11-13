#%%
#Descriptive Statistics
import pandas as pd
import numpy as np

d = pd.read_csv("/home/richard/Documents/Python/Statistics/HW/exam3_data.csv")
df = pd.DataFrame(d)
df

#Print Descriptive Statistics summary
df.describe()

# %%
#Scatter plots
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

df  = pd.read_csv("/home/richard/Documents/Python/Statistics/HW/exam3_data.csv")
# plots all columns against index
df.plot()  

#%%
# scatter plot
df.plot(kind='scatter',x='Age',y='Tumor_Size') 

# %%
# Correlation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

df = pd.read_csv('/home/richard/Documents/Python/Statistics/HW/exam3_data.csv') 
sns.pairplot(df, kind="scatter", diag_kind='auto', hue="Treatment")
plt.show()

# %%
# a cool lil correlation heatmap :)
import seaborn as sns

sns.heatmap(df.corr())
 
# %%
#linear regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

data = pd.read_csv("/home/richard/Documents/Python/Statistics/HW/exam3_data.csv")
data.head()

# Plotting linear regression
plt.style.use('ggplot')

plt.figure(figsize=(11, 8))
plt.scatter(
    data['Age'],
    data['Tumor_Size'],
    c='black'
)
plt.xlabel("Age")
plt.ylabel("Tumor_Size")
plt.show()

# %%
# Equation of a line for Age vs Duration
X = data['Age'].values.reshape(-1,1)
y = data['Tumor_Size'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

# %%
#Anova
data = pd.read_csv("/home/richard/Documents/Python/Statistics/HW/exam3_data.csv")

X = data['Age']
y = data['Tumor_Size']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()

print(est2.summary())

# %%
#Linear regression map with best of fit line
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 

df = pd.read_csv('/home/richard/Documents/Python/Statistics/HW/exam3_data.csv') 
df_binary = df[['Age', 'Tumor_Size']] 

# Taking only the selected two attributes from the dataset
# Renaming the columns for easier writing of the code 
# not really needed... but you know
df_binary.columns = ['Age', 'Tumor_Size']

# Displaying only the 1st  rows along with the column names 
df_binary.head() 

# Some more plotting, change x or y for diffrent models
# can also show best of fit line given 
# the x and y you are comparing.
sns.lmplot(x ="Age", y ="Tumor_Size", data = df_binary, order = 2, ci = None) 

# %%
# Training our data so that it can be modeled

# Seperating the data into independent and dependent variables 
# Converting each dataframe into a numpy array  
# since each dataframe contains only one column 
X = np.array(df_binary['Age']).reshape(-1, 1) 
y = np.array(df_binary['Tumor_Size']).reshape(-1, 1) 
  
# Dropping any rows with Nan values 
df_binary.dropna(inplace = True) 
  
# Splitting the data into training and testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
  
regr = LinearRegression() 
  
regr.fit(X_train, y_train) 
print("Regression Score", regr.score(X_test, y_test))

#%%
# compute R^2 and Adjusted R^2 with statsmodels, 
# by adding intercept manually
import statsmodels.api as sm
X1 = sm.add_constant(X)
result = sm.OLS(y, X1).fit()
result.summary()


# %%
# Data scatter plot of predicted 
# values after some training 
y_pred = regr.predict(X_test) 
plt.xlabel('Age', fontsize=16)
plt.ylabel('Tumor Size', fontsize=16) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, y_pred, color ='y') 
  
plt.show() 

# %%
# This may be useful if the sample 
# size n is small for plot

# Selecting the 100 rows of the data 
df_binary100 = df_binary[:][:100] 
# Plot, this looks the same because 
# the data sample size is already small lol
sns.lmplot(x ="Age", y ="Tumor_Size", data = df_binary100, 
                               order = 2, ci = None) 

# %%
# Trying a lil diffrent traing model
# because the samle size is small,
# this can change the best of fit with every run.
plt.style.use('ggplot')

df_binary100.fillna(method ='ffill', inplace = True) 
  
X = np.array(df_binary100['Age']).reshape(-1, 1) 
y = np.array(df_binary100['Tumor_Size']).reshape(-1, 1)
  
df_binary100.dropna(inplace = True) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
  
regr = LinearRegression() 
regr.fit(X_train, y_train) 
print("Regression Score ", regr.score(X_test, y_test)) 

# Plot the linear model!
y_pred = regr.predict(X_test) 
plt.xlabel('Age', fontsize=16)
plt.ylabel('Tumor Size', fontsize=16) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, y_pred, color ='y')
  
plt.show()

#%%
# compute R^2 and adjusted R^2 with 
# statsmodels, by adding intercept manually
import statsmodels.api as sm
X1 = sm.add_constant(X)
result = sm.OLS(y, X1).fit()
result.summary()

# %%
# R squared and adjusted r squared using 
# Ordinary Least Squares regression
# Using test and predicted y values
from sklearn.metrics import r2_score
import statsmodels.api as sm

r2_score(X_test, y_pred)
r2_adjusted = sm.OLS(X_test, y_pred)
show_r2_adjusted = r2_adjusted.fit()
show_r2_adjusted.summary()

# %%
# Show the pair plots
plt.style.use('ggplot')

sns.pairplot(df[['Race','Age','Treatment','Tumors_Count','Duration','Tumor_Size','Censor']]);

# %%
# Testing out the logistic model
from sklearn.linear_model import LogisticRegression 
plt.style.use('ggplot')

df_binary100.fillna(method ='ffill', inplace = True) 
  
X = np.array(df_binary100['Age']).reshape(-1, 1) 
y = np.array(df_binary100['Tumor_Size']).reshape(-1, 1) 
  
df_binary100.dropna(inplace = True) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
  
regr = LogisticRegression() 
regr.fit(X_train, y_train) 
print(regr.score(X_test, y_test)) 

# Plotting the logistic model!
y_pred = regr.predict(X_test) 
plt.xlabel('Age', fontsize=16)
plt.ylabel('Tumor Size', fontsize=16) 
plt.scatter(X_test, y_test, color ='b') 
plt.plot(X_test, y_pred, color ='y') 
  
plt.show() 

#%%
# compute with statsmodels, by adding intercept manually
import statsmodels.api as sm
X1 = sm.add_constant(X)
result = sm.OLS(y, X1).fit()
result.summary()

# %%
# Testing for diffrent best of fit lines
df_binary = df[['Age', 'Tumor_Size']] 

# Taking only the selected two attributes from the dataset
# Renaming the columns for easier writing of the code 
# not really needed... but you know
df_binary.columns = ['Age', 'Tumor_Size']
sns.lmplot(x ="Age", y ="Tumor_Size", data = df_binary, order = 2, 
           ci = None, fit_reg = True, size = 8)

# %%
# Train data and check for accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

df1 = pd.read_csv('/home/richard/Documents/Python/Statistics/HW/exam3_data.csv')
X = df1.drop("Race", axis = 1)
Y = df1["Race"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2
                                                    , random_state = 42)
# Logistic regression test
clf = LogisticRegression()

clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred))

yhat = clf.predict(X)
SS_Residual = sum((Y-yhat)**2)
SS_Total = sum((Y-np.mean(Y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print (" R^2", r_squared, "Adjusted R^2", adjusted_r_squared)

# %%
# ensemble model data train and accuracy
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, Y_train)
y_pred2 = clf.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred2))

yhat = rfc.predict(X)
SS_Residual = sum((Y-yhat)**2)
SS_Total = sum((Y-np.mean(Y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print (" R^2", r_squared, "Adjusted R^2", adjusted_r_squared)

# %%
# Neighors model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, Y_train)
y_pred3 = knn.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred3))

# Print R^2 and adjusted R^2
yhat = knn.predict(X)
SS_Residual = sum((Y-yhat)**2)
SS_Total = sum((Y-np.mean(Y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print (" R^2", r_squared, "Adjusted R^2", adjusted_r_squared)


# %%
# Decision Tree model
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, Y_train)
y_pred3 = tree_clf.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, y_pred3))

# Print R^2 and adjusted R^2
yhat = tree_clf.predict(X)
SS_Residual = sum((Y-yhat)**2)
SS_Total = sum((Y-np.mean(Y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
print (" R^2", r_squared, "Adjusted R^2", adjusted_r_squared)


# %%
