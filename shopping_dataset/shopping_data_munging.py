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
# %% [markdown]
# data Processing
# QUESTIONS: Is the data in a format 
# that can be passed into an unsupervised learning model?

# %%
# convert categorical into numerical data
# define a function
shopping_df['Card Member'] = shopping_df['Card Member'].apply(lambda a: 1 if a=='Yes' else 0)

# rescale annual income
shopping_df['Annual Income'] = shopping_df['Annual Income'] / 1000

# %%
# rename columns
shopping_df.rename(columns={'Card Member': 'Card_Member', 'Annual Income': 'Annual_Income',
                    'Spending Score (1-100)': 'Spending_Score_(1-100)'}, inplace= True)

shopping_df.head()

# %% [markdown]
# data Transformation 
# QUESTIONS: Can I quickly hand off this data for others to use?
# to readable and user-friendly format
shopping_df.to_csv('shopping_data_cleaned.csv', index = False)