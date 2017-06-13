# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:23:55 2017

@author: admin
"""
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#1. Read File
crmData = pd.read_table('D:\pyTest\CRM.dataset\dataset.tsv', encoding='CP949')
crmData.head()
crmData.describe()
crmData.info()

crmData.values
crmData.columns
crmData.isnull().sum()

#결측치 제거
crmData = crmData.dropna(axis=0)
crmData.isnull().sum()

#2. Data preprocessing
crmData.loc[ crmData['x1'] <= 27, 'x1'] = 0
crmData.loc[(crmData['x1'] > 27) & (crmData['x1'] <= 38), 'x1'] = 1
crmData.loc[(crmData['x1'] > 38) & (crmData['x1'] <= 48), 'x1'] = 2
crmData.loc[(crmData['x1'] > 48) & (crmData['x1'] <= 58), 'x1'] = 3
crmData.loc[(crmData['x1'] > 58) & (crmData['x1'] <= 68), 'x1'] = 4
crmData.loc[(crmData['x1'] > 68) & (crmData['x1'] <= 78), 'x1'] = 5
crmData.loc[(crmData['x1'] > 78) & (crmData['x1'] <= 88), 'x1'] = 6                    
crmData.loc[(crmData['x1'] > 88), 'x1'] = 7

crmData['x1'].hist(bins=70)
(axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

job_mapping = {
    'admin.'   : 0,
    'unknown'  : 1,
    'unemployed' : 2,
    'management' : 3,
    'housemaid'  : 4,
    'entrepreneur' : 5,
    'student.'   : 6,
    'blue-collar'  : 7,
    'self-employed' : 8,
    'retired'   : 9,
    'technician'  : 10,
    'services' : 11,
}
crmData['x2'] = crmData['x2'].map(job_mapping)

marriage_mapping = {
    'single'   : 2,
    'married'  : 1,
    'divorced' : 0
}
crmData['x3'] = crmData['x3'].map(marriage_mapping)

education_mapping = {
    'unknown': 3,
    '고학력' : 2,
    '중학력' : 1,
    '저학력' : 0
}
crmData['x4'] = crmData['x4'].map(education_mapping)

rank_mapping = {
    'yes' : 1,
    'no'  : 0,
}
crmData['x5'] = crmData['x5'].map(rank_mapping)
crmData['x7'] = crmData['x7'].map(rank_mapping)
crmData['x8'] = crmData['x8'].map(rank_mapping)
crmData['y'] = crmData['y'].map(rank_mapping)

stdscaler = StandardScaler()
crmData['x6'] = stdscaler.fit_transform(crmData['x6'])

call_mapping = {
    'unknown': 2,
    '유선' : 1,
    '무선' : 0,
}
crmData['x9'] = crmData['x9'].map(call_mapping)

success_mapping = {
    'other': 3,
    'failure'   : 2,
    'unknown'  : 1,
    'success' : 0
}
crmData['x15'] = crmData['x15'].map(success_mapping)


#stdscaler = StandardScaler()
#crmData['x1'] = stdscaler.fit_transform(crmData['x1'])

crmData = crmData.dropna(axis=0)
crmData.isnull().sum()

X_train = crmData.drop("y", axis=1)
X_train = X_train.drop("x10", axis=1)
X_train = X_train.drop("x11", axis=1)
Y_train = crmData["y"]

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
x_train.shape, y_train.shape
x_test.shape, y_test.shape


random_forest = RandomForestClassifier(n_estimators=100)
#random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, max_features='sqrt', min_samples_split=5)

random_forest.fit(x_train, y_train)

Y_pred_1 = random_forest.predict(x_test)

random_forest.score(x_train, y_train)
random_forest.score(x_test, y_test)

submission = pd.DataFrame({
        "x1": x_test["x1"],
        "x2": x_test["x2"],
        "Survived": Y_pred_1
    })
submission.to_csv('titanic0.csv', index=True)
y_test.value_counts(normalize = True)

submission = pd.DataFrame({
        "x1": X_train["x1"],
        "x2": X_train["x2"],
        "Survived": Y_train
    })
submission.to_csv('titanic3.csv', index=True)

###############################################################################
#데이터 분포 탐색
crmData['y'].value_counts(normalize = True)
crmData['y'][crmData['x3'] == 'married'].value_counts(normalize=True)
crmData['y'][crmData['x3'] == 'single'].value_counts(normalize=True)
crmData['y'][crmData['x6'] < 1000].value_counts(normalize=True)
crmData['y'][crmData['x1'] > 60 ].value_counts(normalize=True)
crmData['y'][crmData['x5'] == 'yes' ].value_counts(normalize=True)
crmData['y'][crmData['x5'] == 'no' ].value_counts(normalize=True)
crmData['y'][crmData['x15'] == 0 ].value_counts(normalize=True)
crmData['x15'][crmData['y'] == 1 ].value_counts(normalize=True)

plt.scatter(crmData['x14'], crmData['y'], s=40)
plt.scatter(crmData['x15'], crmData['y'], s=40)

title_xt = pd.crosstab(crmData['x2'], crmData['y'])
title_xt_pct = title_xt.div(title_xt.sum(1).astype(float), axis=0)

title_xt_pct.plot(kind='bar', 
                  stacked=True, 
                  title='Survival Rate by title')
plt.xlabel('title')
plt.ylabel('Survival Rate')


###############################################################################

df = {'x1':crmData['x1'], 'x2': crmData['x2'], 'x3': crmData['x3'], 'x4': crmData['x4'], 
'x5': crmData['x5'], 'x6': crmData['x6'], 'x7': crmData['x7'], 'x8': crmData['x8'] }
x = pd.DataFrame(df)
x.values
x = x.dropna(axis=0)
x.isnull().sum()

# K Means Cluster
model = KMeans(n_clusters=3)
model.fit(x)

model.labels_

# View the results
# Set the size of the plot
plt.figure(figsize=(14,7))
 
# Create a colormap
colormap = np.array(['red', 'lime', 'black'])
 
# Plot the Original Classifications
#plt.subplot(1, 2, 1)
#plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
#plt.title('Real Classification')
 
# Plot the Models Classifications
#plt.subplot(1, 2, 2)
plt.scatter(x.x2, x.x6,  c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')


