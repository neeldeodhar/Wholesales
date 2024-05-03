#!/usr/bin/env python
# coding: utf-8

# In[55]:


#downloading dataset, importing libraries
import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans


# In[56]:


#reading the dataset, deleting missing value entries
df = pd.read_csv('Wholesale customers data.csv')

df.head()

df.dropna()


# In[57]:


#printing columns list
df.columns


# In[58]:


# NO encoding performed as columns are already numeric
#selecting features
# channel and region columns not selected in selected features
selected_features = df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
display(selected_features)


# In[59]:


# defining x
x_columns = 6

#x = selected_features
x = selected_features.iloc[:, 0:x_columns].values


# In[60]:


#implementing Elbow method in Python
elbow = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters = k)
    kmeanModel.fit(df)
    elbow.append(kmeanModel.inertia_)


# In[61]:


#plotting Elbow method vs values of K
plt.figure(figsize = (16,8))
plt.plot(K, elbow, 'bx-')
plt.xlabel('k values')
plt.ylabel ('elbow')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[62]:


#observation
print ("We can observe that the 'elbow' is the number 2 which is optimal value of k")
print("Now we can use K-Means using n_clusters as number 2")


# In[63]:


#defining a KMEANS model with 2 clusters
model = KMeans(n_clusters =2, random_state = 0)
model.fit(x)  


# In[64]:


#scaling and preprocessing data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(x)


# In[65]:


#defining clusters for model
clusters = model.predict(X)


# In[66]:


#printing model inertia
model.inertia_


# In[67]:


# trying a different optimal value of k, i.e k = 3
model1 = KMeans(n_clusters =3, random_state = 0)
model1.fit(x)  


# In[68]:


#printing model1 inertia
model1.inertia_


# In[69]:


# printing first 10 data samples for model
y = model.fit_predict(x)
print(y[0:10])


# In[70]:


# printing first 10 data samples for model1
y1 = model1.fit_predict(x)
print(y1[0:10])


# In[71]:


#defining centres for model
centres = model.cluster_centers_
print(centres)


# In[72]:


#plotting KMeans cluster between FRESH and FROZEN columns
import matplotlib.pyplot as plt
colors = ['orange', 'blue', 'green']
for i in range(3):
    plt.scatter(x[y1 == i, 0], x[y1 == i, 3], c=colors[i])
plt.scatter(model1.cluster_centers_[:, 0], model1.cluster_centers_[:, 3], color='red', marker='+', s=300)
plt.title('K-Means Clustering')
plt.xlabel('Fresh')
plt.ylabel('Frozen')


# In[73]:


#plotting KMeans cluster between MILK and DETERGENT columns
import matplotlib.pyplot as plt
colors = ['yellow', 'red', 'magenta']
for i in range(3):
    plt.scatter(x[y1 == i, 1], x[y1 == i, 4], c=colors[i])
plt.scatter(model1.cluster_centers_[:, 1], model1.cluster_centers_[:, 4], color='red', marker='+', s=300)
plt.title('K-Means Clustering')
plt.xlabel('Milk')
plt.ylabel('Detergent')


# In[74]:


x


# In[75]:


#HIERARCHICAL clustering part of Agglomerative clustering:

# down loading Scipy library
import scipy.cluster.hierarchy as sch

# generating Agglomerative Clustering, (i.e. a Dendogram)
from sklearn.cluster import AgglomerativeClustering

#SciPy Dendogram (structuring
plt.figure(figsize=(18,10))
plt.title('Dendrogram')
plt.xlabel('selected features')
plt.ylabel('Euclidean distances')
dendrogram = sch.dendrogram(sch.linkage(x, method ='ward'),
                            color_threshold=200, 
                            above_threshold_color='red') 
plt.show()



# In[76]:


# SciKitLearn HIERARCHICAL clustering part of Agglomerative clustering -WITH 3 clusters
modelHC = AgglomerativeClustering(n_clusters = 3, affinity ='euclidean',
                                 linkage ='ward')
yHC = modelHC.fit_predict(x)

plt.scatter(x[:, 0], x[:, 3], c=yHC, cmap="rainbow")
plt.xlabel('Fresh')
plt.ylabel('Frozen')
plt.title("SciKitLearn Agglomerative Clustering")
plt.show()


# In[77]:


# printing first 10 samples with 3 clusters
print(yHC[0:10])


# In[78]:


# SciKitLearn HIERARCHICAL clustering part of Agglomerative clustering -WITH 2 clusters
modelHC1 = AgglomerativeClustering(n_clusters = 2, affinity ='euclidean',
                                 linkage ='ward')
yHC1 = modelHC1.fit_predict(x)

plt.scatter(x[:, 1], x[:, 4], c=yHC1, cmap="rainbow")
plt.xlabel('Milk')
plt.ylabel('Detergent')
plt.title("SciKitLearn Agglomerative Clustering")
plt.show()


# In[79]:


# printing first 10 samples with 2 clusters
print(yHC1[0:10])

