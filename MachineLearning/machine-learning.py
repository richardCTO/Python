#%%
import pandas as pd 

# Path of file to read
iowa_file_path = 'MachineLearning/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# %%
home_data.columns

# %%
y = home_data.SalePrice

# %%
# create the list of features
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF',
                  '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[feature_names]
X.describe()

X.head()

# %%
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 1)

dtr.fit(X, y)

# %%
# Making predictions for the 5 following houses
print('Making predictinos for the 5 following houses')
print(X.head())
print('Here are the predictions ')
dtr.predict(X.head())

# %%
from sklearn.metrics import mean_absolute_error

predicted_home_prices = dtr.predict(X)
mean_absolute_error = (y, predicted_home_prices)
mean_absolute_error

# %%
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

dtr = DecisionTreeRegressor()

dtr.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = dtr.predict(val_X)
mean_absolute_error = (val_y, val_predictions)
mean_absolute_error

# %%
# Mean Absolute Erroe (MSE)
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# %%
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# %%
# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)


home_predictions = forest_model.predict(val_X)
mean_absolute_error(val_y, home_predictions)

# %%
