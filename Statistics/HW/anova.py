#%%
# load packages
import pandas as pd
# load data file
d = pd.read_csv("Data/drug.txt", sep=" ")
# generate a boxplot to see the data distribution by treatments. Using boxplot, we can easily detect the differences 
# between different treatments
d.boxplot(column=['a', 'b', 'c'], grid=False)
#%%
import pandas as pd
# load data file
d = pd.read_csv("Data/drug.txt", sep=" ")

drug_a = d.a.dropna()
drug_b = d.b.dropna()
drug_c = d.c.dropna()

print('Control group\n')
print(drug_a.describe())
print('\nTreatment-1 group\n')
print(drug_b.describe())
print('\nTreatment-2 group\n')
print(drug_c.describe())
#%%
import statsmodels.api as sm
from statsmodels.formula.api import ols

# reshape the d dataframe suitable for statsmodels package 
d_melt = pd.melt(d.reset_index(), id_vars=['index'], value_vars=['a', 'b', 'c'])
# replace column names
d_melt.columns = ['index', 'treatments', 'value']
# Ordinary Least Squares (OLS) model
model = ols('value ~ C(treatments)', data=d_melt).fit()
# replace column names
d_melt.columns = ['drugs', 'between', 'value']
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table

# %%
# load packages
# get ANOVA table as R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols

# reshape the d dataframe suitable for statsmodels package 
d_melt = pd.melt(d.reset_index(), id_vars=['index'], value_vars=['a', 'b', 'c'])

# replace column names
d_melt.columns = ['drugs', 'between', 'value']

# Ordinary Least Squares (OLS) model
model = ols('value ~ C(between)', data=d_melt).fit()
print(model.summary())

# %%
