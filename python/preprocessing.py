import pandas as pd
import sklearn.cluster as cluster
import numpy as np
import re
from sklearn.preprocessing import Imputer
import knnimpute
from sklearn.preprocessing import StandardScaler
#This file is for transforming the csvs into a dataframe that can be called later

data_dir="../data/"

df = pd.read_csv(data_dir+"77_cancer_proteomes_CPTAC_itraq.csv", header=0,index_col=0)
patient_info = pd.read_csv(data_dir+"clinical_data_breast_cancer.csv", header=0,index_col=0)
pam50 = pd.read_csv(data_dir+"PAM50_proteins.csv", header=0,index_col=0)

df.drop(['gene_symbol','gene_name'],axis=1,inplace=True)

df.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)

df=df.transpose()

## Drop clinical entries for samples not in our protein data set
patient_info = patient_info.loc[[x for x in patient_info.index.tolist() if x in df.index],:]

## Add clinical meta data to our protein data set, note: all numerical features for analysis start with NP_ or XP_
preprocessed = df.merge(patient_info,left_index=True,right_index=True)

## Numerical data for the algorithm, NP_xx/XP_xx are protein identifiers from RefSeq database
preprocessed_numeric = preprocessed.loc[:,[x for x in preprocessed.columns if bool(re.search("NP_|XP_",x)) == True]]

## Select only the PAM50 proteins - known panel of genes used for breast cancer subtype prediction
preprocessed_numerical_p50 = preprocessed_numeric.ix[:,preprocessed_numeric.columns.isin(pam50['RefSeqProteinID'])]

'''
Removing rows with 20% missing values, standard deviations less than
0.4 at log-2, and standard scaling will be done. Afterwards the processing will branch
into two methods of feature reduction: filling missing values with knn or medians. I'll
leave PCA or other feature reductions for the algorithms section.
'''


##Remove rows with more than 20% missing values, STD < 0.4, Standard Scaler
preprocessed_nan_thresh=preprocessed_numerical_p50.dropna(thresh=0.8*len(preprocessed_numerical_p50), axis=1)
preprocessed_log2 = np.log2(preprocessed_nan_thresh)
preprocessed_scaled = StandardScaler().fit_transform(preprocessed_log2)
threshold = 0.4
preprocessed_scaled.drop(preprocessed_scaled.std()[preprocessed_scaled.std() < threshold].index.values, axis=1, inplace=True)
## Impute missing values with median
imputer = Imputer(missing_values='NaN', strategy='median', axis=1)
imputer = imputer.fit(preprocessed_scaled)
median_filled = imputer.transform(preprocessed_scaled)
median_filled.to_csv(data_dir+"median_filled.csv")

##Impute with knn
mask=np.isnan(preprocessed_scaled)
knn_filled = knnimpute.knn_impute_optimistic(preprocessed_scaled, mask, k=3)   
knn_filled.to_csv(data_dir+"knn_filled.csv")
