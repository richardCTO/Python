#%%
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd

# import data set and print first 10 lines
nba_file_path = pd.read_csv('DataVisualization/nba-players-data/all_seasons.csv')
nba_data = pd.DataFrame(nba_file_path)
nba_data.head(n = 10)

# %%
df_binary = nba_data[['player_height', 'pts']] 

# use only these two colums for plots
df_binary.columns = ['players_height', 'pts']
# Prints the first 10 lines just for players_height and pts
df_binary.head(n = 10)

#%%
# makes the graphs look pretty :)
plt.style.use('ggplot')

# Plot players_height vs pts with a regression line
sns.lmplot(x ="players_height", y ="pts", data = df_binary, order = 2, ci = None, 
           size=10) 

# %%
