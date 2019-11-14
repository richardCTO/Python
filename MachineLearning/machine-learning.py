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
