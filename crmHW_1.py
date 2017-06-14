# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:26:36 2017

@author: Seo, Jeong yeon
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


#1. Read File
###.Please input path for the 'dataset.tsv' ###
input_file = 'D:\dataAnalysis_python\dataset.tsv'
crmData = pd.read_table(input_file, encoding='CP949')


#2. Data information
crmData.head()
crmData.describe()
crmData.info()
crmData.values
crmData.columns
crmData.isnull().sum()

#3. Data preprocessing
#stdscaler = StandardScaler()
#crmData['x1'] = stdscaler.fit_transform(crmData['x1'])

crmData.loc[ crmData['x1'] <= 27, 'x1'] = 0
crmData.loc[(crmData['x1'] > 27) & (crmData['x1'] <= 38), 'x1'] = 1
crmData.loc[(crmData['x1'] > 38) & (crmData['x1'] <= 48), 'x1'] = 2
crmData.loc[(crmData['x1'] > 48) & (crmData['x1'] <= 58), 'x1'] = 3
crmData.loc[(crmData['x1'] > 58) & (crmData['x1'] <= 68), 'x1'] = 4
crmData.loc[(crmData['x1'] > 68) & (crmData['x1'] <= 78), 'x1'] = 5
crmData.loc[(crmData['x1'] > 78) & (crmData['x1'] <= 88), 'x1'] = 6                    
crmData.loc[(crmData['x1'] > 88), 'x1'] = 7

#crmData['x1'].hist(bins=70)

rank_mapping = {
    'yes' : 1,
    'no'  : 0,
}
crmData['x5'] = crmData['x5'].map(rank_mapping)
crmData['x7'] = crmData['x7'].map(rank_mapping)
crmData['x8'] = crmData['x8'].map(rank_mapping)

education_mapping = {
    '고학력' : 2,
    '중학력' : 1,
    '저학력' : 0
}
crmData['x4'] = crmData['x4'].map(education_mapping)

marriage_mapping = {
    'single'   : 2,
    'married'  : 1,
    'divorced' : 0
}
crmData['x3'] = crmData['x3'].map(marriage_mapping)

job_mapping = {
    'admin.'        : 0,
    'unknown'       : 1,
    'unemployed'    : 2,
    'management'    : 3,
    'housemaid'     : 4,
    'entrepreneur'  : 5,
    'student.'      : 6,
    'blue-collar'   : 7,
    'self-employed' : 8,
    'retired'       : 9,
    'technician'    : 10,
    'services'      : 11,
}
crmData['x2'] = crmData['x2'].map(job_mapping)

stdscaler = StandardScaler()
crmData['x6'] = stdscaler.fit_transform(crmData['x6'])


# 4. 고객정보 X1~X8 속성의 데이터 구성
df = {'x1':crmData['x1'], 'x2': crmData['x2'], 'x3': crmData['x3'], 'x4': crmData['x4'], 
'x5': crmData['x5'], 'x6': crmData['x6'], 'x7': crmData['x7'], 'x8': crmData['x8'] }

x = pd.DataFrame(df)

# 4-1. 결측치 제거
x = x.dropna(axis=0)
x.isnull().sum()


# 5. clustering: K-means알고리즘
# 5-1. cluster 갯수 선정
### data가 다른 군집보다 자신이 속한 군집에 얼마나 가까운지를 판단하는 척도인 실루엣점수를 이용하여 클러스터 갯수를 결정하였습니다.
scores = []
values = np.arange(2, 7)
n_clusters = 0
maxScore = 0

for num_clusters in values:
    kmeans = KMeans(init = 'k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(x)

    score = metrics.silhouette_score(x, kmeans.labels_, metric = 'euclidean', sample_size=len(x))

    print('\n********************Number of clusters in input data = ', num_clusters,'********************')
    print('\n********************Silhouette score = ', score,'********************')
    scores.append(score)
    
    if score > maxScore:
        maxScore = score
        n_clusters = num_clusters
    else:
        pass

plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')

print('\n********************Silhouette score = ', maxScore, '********************')
print('\n********************Number of clusters for K Means Cluster = ', n_clusters, '********************')


# 5-2. K Means Cluster 적용
model = KMeans(n_clusters)
model.fit(x)
model.labels_

# 5-3. View the results
plt.figure(figsize=(14,7))
 
# Create a colormap
colormap = np.array(['red', 'lime', 'black', 'pink'])
 
 
# Plot the Models Classifications
plt.subplot(1, 2, 1)
plt.scatter(x.x1, x.x8,  c=colormap[model.labels_], s=40)
plt.subplot(1, 2, 2)
plt.scatter(x.x7, x.x1,  c=colormap[model.labels_], s=40)