#%%
# Time Series testing
import pandas as pd 

df = pd.read_csv("Time Series/aud_usd_data.csv")
df.head(n=10)

# %%
type(df.Date[0])

# %%

