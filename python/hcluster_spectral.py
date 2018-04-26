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
df=pd.read_csv(data_dir+"median_filled.csv", header=0, index_col=0)
