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
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage

data_dir="../data/"
df=pd.read_csv(data_dir+"median_filled.csv", header=0, index_col=0)

linkage=['ward', 'average', 'complete']

n_clusters=4
clustering_ward = AgglomerativeClustering(linkage=linkage[0], n_clusters=n_clusters,compute_full_tree=True)
clustering_avg = AgglomerativeClustering(linkage=linkage[1], n_clusters=n_clusters,compute_full_tree=True)
clustering_complete = AgglomerativeClustering(linkage=linkage[2], n_clusters=n_clusters,compute_full_tree=True)

clustering_ward.fit(df)
clustering_avg.fit(df)
clustering_complete.fit(df)

#plt.gcf().subplots_adjust(bottom=0.4)

g=sns.clustermap(df,metric="correlation",cmap="RdYlGn")
plt.savefig("../plots/clustermap.png")
plt.show()
