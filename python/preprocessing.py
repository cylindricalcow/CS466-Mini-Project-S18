import pandas as pd
import sklearn.cluster as cluster
import numpy as np
import re
from sklearn.preprocessing import Imputer
import knnimpute
from sklearn.preprocessing import StandardScaler
#This file is for transforming the csvs into a dataframe that can be called later

def median_rows(x):
    #n=len(x)
    new_list=[]
    new_list2=[]
    for val in x:
        if val!=-9999:
            new_list.append(val)
    med=np.median(new_list)
    for val in x:
        if val!=-9999:
            new_list2.append(val)
        else:
            new_list2.append(med)
    return new_list2         
data_dir="../data/"

df = pd.read_csv(data_dir+"77_cancer_proteomes_CPTAC_itraq.csv", header=0,index_col=0)
patient_info = pd.read_csv(data_dir+"clinical_data_breast_cancer.csv", header=0,index_col=0)
pam50 = pd.read_csv(data_dir+"PAM50_proteins.csv", header=0,index_col=0)
df.fillna(-9999, inplace=True)

df.drop(['gene_symbol','gene_name'],axis=1,inplace=True)

df.rename(columns=lambda x: "TCGA-%s" % (re.split('[_|-|.]',x)[0]) if bool(re.search("TCGA",x)) is True else x,inplace=True)

df=df.transpose()

keep_all=True
if keep_all:
    print(len(df.columns),len(df.index))
    df2=df[(df==-9999).sum(axis=1)/len(df.columns) <= 0.20]

    #median_nan_thresh=preprocessed_numerical_p50.drop(thresh=0.8*len(preprocessed_numerical_p50), axis=1)
    #median_nan_thresh.fillna(median_nan_thresh.mean(), inplace=True)
    #median_nan_thresh=preprocessed_numerical_p50.replace(-9999, preprocessed_numerical_p50.median(), axis=1)
    median_nan_thresh=df2.apply(median_rows)
    #print(median_nan_thresh)


    X=StandardScaler().fit_transform(median_nan_thresh)
    median_scaled = pd.DataFrame(X,index=median_nan_thresh.index, columns=median_nan_thresh.columns)
    threshold = 0.4
    median_scaled.drop(median_scaled.std()[median_scaled.std() < threshold].index.values, axis=1, inplace=True)
    median_scaled.to_csv(data_dir+"median_filled_all.csv")
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

## Impute missing values with median
'''
imputer = Imputer(missing_values='NaN', strategy='median', axis=1)
imputer = imputer.fit(preprocessed_numerical_p50)
median_filled = imputer.transform(preprocessed_numerical_p50)
'''

preprocessed_numerical_p50=preprocessed_numerical_p50[(preprocessed_numerical_p50==-9999).sum(axis=1)/len(preprocessed_numerical_p50.columns) <= 0.20]

#median_nan_thresh=preprocessed_numerical_p50.drop(thresh=0.8*len(preprocessed_numerical_p50), axis=1)
#median_nan_thresh.fillna(median_nan_thresh.mean(), inplace=True)
#median_nan_thresh=preprocessed_numerical_p50.replace(-9999, preprocessed_numerical_p50.median(), axis=1)
median_nan_thresh=preprocessed_numerical_p50.apply(median_rows)
#print(median_nan_thresh)
'''
for a in median_nan_thresh:
    if np.isnan(a) == True:
        print("whoops")
'''
#median_log2 = median_nan_thresh.applymap(np.log2)
#print(median_log2)
####REMOVED log2 part since it created NaN values
X=StandardScaler().fit_transform(median_nan_thresh)
median_scaled = pd.DataFrame(X,index=median_nan_thresh.index, columns=median_nan_thresh.columns)
threshold = 0.4
median_scaled.drop(median_scaled.std()[median_scaled.std() < threshold].index.values, axis=1, inplace=True)
median_scaled.to_csv(data_dir+"median_filled.csv")

#######################Currently broken
##Impute with knn
preprocessed_numerical_p50=preprocessed_numerical_p50.replace(-9999, np.nan)
print(preprocessed_numerical_p50)
mask=np.isnan(preprocessed_numerical_p50)
#knn_filled = knnimpute.knn_impute_optimistic(preprocessed_numerical_p50, mask, k=3)
knn_filled = knnimpute.knn_impute_with_argpartition(preprocessed_numerical_p50, mask, k=3)
knn_filled= pd.DataFrame(knn_filled, index=preprocessed_numerical_p50.index, columns=preprocessed_numerical_p50.columns)

knn_log2 = knn_filled.applymap(np.log2)
knn_scaled = pd.DataFrame(StandardScaler().fit_transform(knn_log2),index=knn_log2.index, columns=knn_log2.columns)
threshold = 0.4
knn_scaled.drop(knn_scaled.std()[knn_scaled.std() < threshold].index.values, axis=1, inplace=True)




  
knn_scaled.to_csv(data_dir+"knn_filled.csv")
