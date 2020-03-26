# %%
import pandas as pd 


# %%
iris_df = pd.read_csv('iris.csv')
iris_df.head()

# %%
# data selection and process(format, clean, sampling)
new_iris_df = iris_df.drop('class', axis = 1)
new_iris_df = new_iris_df[['sepal_length', 'petal_length',
                           'sepal_width', 'petal_width']]
new_iris_df.head()
# %%
# data transform
new_iris_df.to_csv('new_iris.csv', index = False)

# %%
