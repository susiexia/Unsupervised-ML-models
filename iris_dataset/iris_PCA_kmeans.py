# %% [markdown]

# Use PCA(principal component analysis) to reduce 4 variables (dimension) 
# into 2 input features. Then use kmeans model to cluster iris datasets

# %%
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import hvplot.pandas
import plotly.express as px 

# %%
df = pd.read_csv('new_iris_data.csv')
df.head(n=-5)

# %%
# STEP 1: standardize 4 features
iris_ndarray = StandardScaler().fit_transform(df)
type(iris_ndarray)
iris_ndarray[:5, :]

# %%
# STEP 2: PCA model
pca = PCA(n_components=2, random_state=0)
iris_pca = pca.fit_transform(iris_ndarray)
ratio = pca.explained_variance_ratio_
ratio # the information attributed to each principal component features
# %%
# STEP 3: build a DataFrame for pca_ndarray
iris_pca_df = pd.DataFrame(data=iris_pca, 
                    columns=['Principal_Component_1','Principal_Component_2'])
iris_pca_df.head(n=-5)


# %%
# use elbow curve to find best value of K
inertia_list = []
k_list = list(range(1,11))

for k in k_list:
    km = KMeans(n_clusters=k, random_state=0)
    km.fit(iris_pca_df)
    inertia_list.append(km.inertia_)
elbow_df = pd.DataFrame({'k': k_list, 'inertia': inertia_list})

# viz
obj = elbow_df.hvplot.line(x='k', y='inertia', xticks = k_list, title= 'Elbow Curve')
hvplot.show(obj)
# %%
# KMeans model with K = 3
prediction = KMeans(n_clusters=3, random_state=0).fit_predict(iris_pca_df)

iris_pca_df['class'] = prediction
iris_pca_df.head(n=-5)

# %%
# viz
obj = iris_pca_df.hvplot.scatter(x='Principal_Component_1', y='Principal_Component_2',
                                 by = 'class' )
hvplot.show(obj)

# %%
