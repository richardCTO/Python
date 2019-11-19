#%%
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd

# import data set and print first 10 lines
nba_file_path = pd.read_csv('DataVisualization/nba-players-data/all_seasons.csv', index_col=0)
nba_data = pd.DataFrame(nba_file_path)
nba_data.head(n = 10)

#%%
# Check data types and if any records are missing
nba_data.info()

#%%
# Descriptive statistics for the data set
nba_data.describe()

# %%
df_binary = nba_data[['player_height', 'pts']] 

# use only these two colums for plots
df_binary.columns = ['players_height', 'pts']
# Prints the first 10 lines just for players_height and pts
df_binary.head(n = 10)

#%%
import seaborn as sns
sns.set()
# makes the graphs look pretty :)
plt.style.use('ggplot')

# Plot players_height vs pts with a regression line
sns.lmplot(x ="players_height", y ="pts", data = df_binary, size=10) 

plt.title("Height vs Points Scored")
plt.ylabel('Height (cm)')
plt.xlabel('Points Scored')

plt.show()

# %%

# Data training
import numpy as np 
from sklearn.model_selection import train_test_split

X = np.array(nba_data.drop(columns= ['player_name' ,'team_abbreviation','college', 'season',
                            'country', 'draft_year', 'draft_round','draft_number'], axis=1))
y = np.array(nba_data['pts'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
df_binary.dropna(inplace = True) 

# %%
#Linear Regression model
import numpy as np 
from sklearn import metrics
from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

lreg.fit(X_train, y_train)
y_pred = lreg.predict(X_test)

print("Accuracy", metrics.accuracy_score(y, y_pred))

# %%
