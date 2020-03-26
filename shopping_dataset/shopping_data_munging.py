# %%
import pandas as pd 

# %%
# data loading
shopping_df = pd.read_csv('shopping_data.csv')
shopping_df.head()

# %% [markdown]
# data selection 
# QUESTIONS: What data is available? What type? 
# What data is missing? What data can be removed?


# %%
shopping_df.columns

# %%
shopping_df.dtypes

# %%
for col in shopping_df.columns:
    print(f"Cloumn {col} has {shopping_df[col].isnull().sum()} Null values")
# %%
shopping_df.duplicated().sum()
# %%
shopping_df = shopping_df.dropna()

shopping_df.drop(columns=['CustomerID'], inplace=True)

shopping_df.head()
# %%
