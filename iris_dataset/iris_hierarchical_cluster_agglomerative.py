# %% [markdown]
# AgglomerativeClustering: starts by declaring each point 
# with its own cluster, then merges with most similar clusters until
# reach a declared stopping point

# Use dendrogram to show the cutoff line for determining K

# %%
import pandas as pd 
import plotly.figure_factory as ff 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

import hvplot.pandas

# %%
df = pd.read_csv('new_iris_data.csv')

# %%
# Standardized first
iris_scale = StandardScaler().fit_transform(df)

# apply PCA to reduce to 2 pricipal components
pca_model = PCA(n_components=2, random_state=0)
iris_pca = pca_model.fit_transform(iris_scale)
print(pca_model.explained_variance_ratio_)
# %%
iris_pca_df = pd.DataFrame(data=iris_pca, columns=['PC_1','PC_2'])
iris_pca_df.head(-5)

# %%
# creating a dendrogram using plotly.figure_factory
fig = ff.create_dendrogram(iris_pca_df, color_threshold=0)
fig.update_layout(width = 800, height = 500)
fig.show()
# the higher the horizontal lines, the less similarity there is between the clusters.

# %%
agg = AgglomerativeClustering(n_clusters=3)
model = agg.fit(iris_pca_df)

iris_pca_df['class'] = model.labels_
iris_pca_df.head(-5)

# %%
# data viz
obj = iris_pca_df.hvplot.scatter(x='PC_1', y='PC_2', 
                            by='class', hover_cols = ['class'])

hvplot.show(obj)

# %%
