# %% [markdown]
# K-means: an unsupervised learning algorithm used to identify and solve clustering issues.
# A centroid is a data point that is the arithmetic mean position of all the points on a cluster


# %%
import pandas as pd 
from sklearn.cluster import KMeans
import plotly.express as px
import hvplot.pandas

# %%
# cleaned dataset loading
iris_df = pd.read_csv('new_iris.csv')
iris_df.head(3)
# %%
# Init the K starting centroids, with K =3
model = KMeans(n_clusters=3, random_state= 5)

# data points assigned to nearest centroid
# (When the model is being trained (fit the data), the K-means algorithm will iteratively look for the best centroid for each of the K clusters)
model.fit(iris_df)

# group data points
predictions = model.predict(iris_df)
predictions

# %%
# get info about three centroid coordinates
model.cluster_centers_
# %%
# add predictions outcome back to original dataframe
iris_df['class'] = model.labels_

#iris_df['class'] = predictions
iris_df.head()

# %% [markdown]
# data Visualization of results

# %%
# use hvplot to plot 2D scatter, use bokeh to viz (for web browsers)
import holoviews as hv
hv.extension('bokeh')
iris_df.hvplot.scatter(x='petal_width', y='sepal_length', by ='class')
hvplot.show(iris_df.hvplot.scatter(x='petal_width', y='sepal_length', by ='class'))
# %%
# use plotly.express library
fig = px.scatter_3d(iris_df, x='petal_width', y='sepal_length', z='petal_length',
                    color='class', symbol='class', size='sepal_width', width=800)
fig.update_layout(legend = dict(x=0, y=1)) # update the legend position to legend = {'x': 0, 'y':1}
fig.show()

# %%


# %%
