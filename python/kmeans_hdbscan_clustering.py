import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import scipy
from sklearn import metrics
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import hdbscan as hd

data_dir="../data/"
#df=pd.read_csv(data_dir+"median_filled.csv", header=0, index_col=0)
df=pd.read_csv(data_dir+"median_filled_all.csv", header=0, index_col=0)
n=len(df.columns)
def find_ncomponents(arr, val=0.9, n=n):
    diffs=np.abs(arr-val)
    closest=diffs.argmin()
    
    return np.arange(n)[closest]

def cluster(X,method, min_cluster_size):
        '''
        Fits HBSCAN clustering on array. HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications with Noise. 
        Performs DBSCAN over varying epsilon values and integrates the result to find a clustering 
        that gives the best stability over epsilon. This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN), 
        and be more robust to parameter selection. 
    
        Parameters
        ----------
        array: A numpy array
        
                      
        Returns
        -------
        An array with the labels for which clusters the particles belong to. '-1' means noise which should be treated as an 
        individual.
        '''
        if method=='HDBSCAN':
            clusterer = hd.HDBSCAN(min_cluster_size=min_cluster_size) 
        else:
            clusterer=KMeans(n_clusters=min_cluster_size)
        clusterer.fit(X)
        if method=='HDBSCAN':
            probs=clusterer.probabilities_ 
            return max(clusterer.labels_+1),clusterer
        else:
            inertia=clusterer.inertia_
            centers=clusterer.cluster_centers_       
            return inertia,clusterer

def elbow_graph(X,method,start=2, end=15):
        #make an elbow graph that allows user to set min_cluster_size. Min_cluster_size should
        #be the first one with a dramatic drop from ~10^2 to ~10^1.
        min_size=[]
        number_of_clusters=[]
        for i in range(start,end+1):
            min_size.append(i)
           
            number_of_clusters.append(cluster(X,method,min_cluster_size=i)[0])
        _, ax=plt.subplots()
        if method=="HDBSCAN":
            ax.set(xlabel='Minimum Number in Cluster', ylabel='Number of clusters', title='The elbow method')
        else:
            ax.set(ylabel='Inertia', xlabel='Number of clusters', title='The elbow method')
        plt.xticks(np.arange(start,end, 1))    
        plt.plot(min_size,number_of_clusters)
        plt.show()

#Get component that corresponds to cumsum=0.9
pca = PCA().fit(df)
arr=np.cumsum(pca.explained_variance_ratio_)

best_n=find_ncomponents(arr,0.9,n)
print(best_n)

plt.plot(arr)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
print(df.head())
pca_best = PCA(n_components=best_n)
X = pca_best.fit(df).transform(df)
elbow_graph(X, 'HDBSCAN',2, 12)
pca2= PCA(n_components=2)
X2 = pca2.fit(df).transform(df)

elbow_graph(X2, 'HDBSCAN',2, 12)

hdbscan_cluster_pca=cluster(X2,"HDBSCAN", min_cluster_size=3)[1]

sc=plt.scatter(X2[:,0],X2[:,1], s=8, alpha=1,c=hdbscan_cluster_pca.labels_)
clb = plt.colorbar(sc)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("HDBSCAN Clustering")
#plt.savefig("../plots/hdbscan_pca2.png")
plt.savefig("../plots/hdbscan_pca2_all.png")
plt.show()

elbow_graph(X2, 'Kmeans',2, 12)

kmeans_cluster_pca=cluster(X2,"Kmeans", min_cluster_size=4)[1]

sc=plt.scatter(X2[:,0],X2[:,1], s=8, alpha=1,c=kmeans_cluster_pca.labels_)
clb = plt.colorbar(sc)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("Kmeans Clustering")
#plt.savefig("../plots/kmeans_pca2.png")
plt.savefig("../plots/kmeans_pca2_all.png")
plt.show()

