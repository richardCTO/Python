#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

pokemon_data = pd.read_csv("Modeling/data/pokemon/pokemon.csv")
pokemon_data.head(10)
pokemon_data.tail()

# %%
# prints out all the columns for the data set
pokemon_data.columns

# %%
# change types of columns from int to float to avoid future warnings
for col in pokemon_data.columns:
    if pokemon_data[col].dtype == int:
        pokemon_data[col] = pokemon_data[col].astype(float)

# %%
# we want to predict if pokemon is legendary or not
pokemon_data['isLegendary'].value_counts()

# %%
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(pokemon_data, test_size=0.2, random_state=42)

# %%
# convert into array
def get_arrays(pokemon_df):
    X = np.array(pokemon_data[['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def']])
    y = np.array(pokemon_data['isLegendary'])
    
    return X, y

X_train, y_train = get_arrays(df_train)
X_test, y_test = get_arrays(df_test)

X_train.shape, y_train.shape

# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# %%
# train and predict
model = pipeline.fit(X_train, y_train)
model.predict(X_train)[:5]

# %%
# confusion matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train, model.predict(X_train))

# %%
# accuracy score
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train, model.predict(X_train))

# %%
