#%%
import pandas as pd


datafile = 'https://vincentarelbundock.github.io/Rdatasets/csv/datasets/PlantGrowth.csv'
df = pd.read_csv(datafile)

#Create a boxplot
df.boxplot('weight', by='group', figsize=(12, 8))

ctrl = df['weight'][df.group == 'ctrl']

grps = pd.unique(df.group.values)
d_data = {grp:df['weight'][df.group == grp] for grp in grps}

k = len(pd.unique(df.group))  # number of conditions
N = len(df.values)  # conditions times participants
n = df.groupby('group').size()[0] #Participants in each condition

# %%