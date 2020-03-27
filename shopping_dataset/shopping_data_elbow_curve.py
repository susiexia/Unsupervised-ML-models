# %% [markdown]
# use elbow curve to determine the best guess K value for unsupervised ML
# %%
import pandas as pd 
from sklearn.cluster import KMeans

import hvplot.pandas
import plotly.express as px

# %%
shopping_df = pd.read_csv('shopping_data_cleaned.csv')
shopping_df.head(3)


# %%
inertia = list()
k_value = list(range(1,11)) # list integers from 1 to 10

# calculate the inertia for the range of K values
for k in k_value:
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(shopping_df)

    inertia.append(model.inertia_)

# create a DF for plotting
elbow_df = pd.DataFrame({'k':k_value, 'inertia': inertia})
elbow_df.head()


# %%
obj = elbow_df.hvplot.line(x='k', y='inertia', 
                           xticks = k_value, title = 'Elbow Curve')
hvplot.show(obj)

# Either K values for points 5 or 6 could be considered the elbow
# %%
# create a function to test k = 5 and 6

def test_clusters(df, k):
    km_model = KMeans(n_clusters=k, random_state=0)
    km_model.fit(df)
    df['class'] = km_model.labels_

    return df

# call function by passing k = 5 and 6
five_clusters = test_clusters(shopping_df, 5)
five_clusters.head()
# %%
# plot 3D scatter with x="Annual Income", y="Spending Score (1-100)" and z="Age"
fig = px.scatter_3d(five_clusters, x="Age",
                    y="Spending_Score_(1-100)",
                    z="Annual_Income",
                    color="class",
                    symbol="class",
                    width=800)
fig.update_layout(legend = {'x': 0, 'y': 1})
fig.show()
# %%
six_clusters = test_clusters(shopping_df, 6)
six_clusters.head()


# %%
# plot 3D scatter with x="Annual Income", y="Spending Score (1-100)" and z="Age"
fig = px.scatter_3d(six_clusters, x="Age",
                    y="Spending_Score_(1-100)",
                    z="Annual_Income",
                    color="class",
                    symbol="class",
                    width=800)
fig.update_layout(legend = {'x': 0, 'y': 1})
fig.show()

# %% [markdown]
# One might conclude that six groups would be most useful because they could be broken down like so:

# Cluster 0: medium income, low annual spend
# Cluster 1: low income, low annual spend
# Cluster 2: high income, low annual spend
# Cluster 3: low income, high annual spend
# Cluster 4: medium income, high annual spend
# Cluster 5: very high income, high annual spend