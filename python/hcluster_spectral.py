import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import scipy
from sklearn import metrics
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pickle
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
def median_rows(x):
    #n=len(x)
    new_list=[]
    new_list2=[]
    for val in x:
        if val!=float("-inf"):
            new_list.append(val)
    med=np.median(new_list)
    for val in x:
        if val!=float("-inf"):
            new_list2.append(val)
        else:
            new_list2.append(med)
    return new_list2 

data_dir="../data/"
df=pd.read_csv(data_dir+"median_filled.csv", header=0, index_col=0)
'''
df_norm=MinMaxScaler().fit_transform(df)
df_ = np.log2(df_norm)
#Find indicies that you need to replace
inds = np.where(np.isnan(df_))
col_mean = np.nanmean(df_, axis=0)
#Place column means in the indices. Align the arrays using take
df_[inds] = np.take(col_mean, inds[1])

df=pd.DataFrame(df_, index=df.index, columns=df.columns)
df=df.apply(median_rows)
seaborn_map = sns.clustermap(df,cmap="RdYlGn")
# now we keep this clustering, but recreate our data to fit the above clustering, with our minor
# index below the major index (you can think of transcript levels under gene levels if you are
# a biologist)

# recreate our dendrogram, this is undocumented and probably a hack but it works
seaborn_map.dendrogram_col.plot(seaborn_map.ax_col_dendrogram)
 
plt.show()

'''

linkage=['ward', 'average', 'complete']

n_clusters=4
clustering_ward = AgglomerativeClustering(linkage=linkage[0], n_clusters=n_clusters,compute_full_tree=True)
clustering_avg = AgglomerativeClustering(linkage=linkage[1], n_clusters=n_clusters,compute_full_tree=True)
clustering_complete = AgglomerativeClustering(linkage=linkage[2], n_clusters=n_clusters,compute_full_tree=True)

clustering_ward.fit(df)
clustering_avg.fit(df)
clustering_complete.fit(df)
'''
#plt.gcf().subplots_adjust(bottom=0.4)

g=sns.clustermap(df,cmap="RdYlGn") 
plt.savefig("../plots/clustermap.png")
plt.show()
'''
def spectral_cluster(X,n_clusters):
    clusterer=SpectralClustering(n_clusters=n_clusters,eigen_solver='arpack',affinity='nearest_neighbors')#,n_neighbors=n_neighbors)
    clusterer.fit(X)
    labels= clusterer.labels_
    return labels, clusterer
'''
n_samples=df.shape[0]
clusterer=SpectralClustering(n_clusters=8,eigen_solver='arpack',affinity='nearest_neighbors')
clusterer.fit(df)
labels= clusterer.labels_
W = clusterer.affinity_matrix_
G=nx.from_scipy_sparse_matrix(W)
       
d=np.reshape(np.array(W.sum(axis=0)),n_samples)
D=np.diag(d)
L=D-W
lmbda, U = np.linalg.eigh(L)
num=10
lmbds=lmbda[:num]
diffs=[]
i_s=[]
for i in range(num-1):
    diffs.append(lmbds[i+1]-lmbds[i])
    i_s.append(i+1)
print(diffs)    
print(np.argsort(diffs)[::-1] )
plt.scatter(i_s,diffs)
plt.xlabel("k clusters")
plt.ylabel("Difference in Eigenvalues")
plt.savefig("../plots/Spectral Eigenvalue Difference.png")
plt.close()
plt.scatter(np.arange(10),lmbds)
plt.xlabel("k clusters")
plt.ylabel("Eigenvalue")
plt.savefig("../plots/Spectral Eigenvalue.png") 
plt.show()
'''
clusterer_spetral=spectral_cluster(df,4)[0]
pca2= PCA(n_components=2)
X = pca2.fit(df).transform(df)
'''
sc=plt.scatter(X[:,0],X[:,1], s=8, alpha=1,c=clusterer_spetral)
clb = plt.colorbar(sc)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Spectral Clustering")
plt.savefig("../plots/spectral_pca2.png")
plt.show()
'''
sc=plt.scatter(X[:,0],X[:,1], s=8, alpha=1,c=clustering_ward.labels_)
clb = plt.colorbar(sc)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Ward Agglomerative Clustering")
plt.savefig("../plots/ward_agg_pca2.png")
plt.show()

sc=plt.scatter(X[:,0],X[:,1], s=8, alpha=1,c=clustering_avg.labels_)
clb = plt.colorbar(sc)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Average Agglomerative Clustering")
plt.savefig("../plots/avg_agg_pca2.png")
plt.show()

sc=plt.scatter(X[:,0],X[:,1], s=8, alpha=1,c=clustering_complete.labels_)
clb = plt.colorbar(sc)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Complete Agglomerative Clustering")
plt.savefig("../plots/complete_agg_pca2.png")
plt.show()
