# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 08:56:04 2017

@author: admin
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
from sklearn.linear_model import LinearRegression

#1. Read File
###.Please input path for the 'dataset.tsv' ###
input_file = 'D:\pyAnal\M_BASE.csv'
aData = pd.read_table(input_file, encoding='CP949', delimiter=',')

#2. Data information
aData.head()
aData.describe()
aData.info()
#aData.values
aData.columns
aData.isnull().sum()


plt.plot(aData['FIC220'], label = 'FIC220')
plt.plot(aData['FIC201'], label = 'FIC201')
plt.plot(aData['FIC221'], label = 'FIC221')
plt.plot(aData['FIC220']+aData['FIC201']+aData['FIC221'])
plt.legend()
plt.show()

#data preprocessing
aData = aData.dropna(axis=0)
aData.isnull().sum()
len(aData.FIC201)
aData.FIC201[49998]


for i in range(len(aData.FIC201)):
    if aData.FIC201[i] < 10:
       aData.FIC201[i] = '0'
      # print(job[i])
    else:
        pass
    i += 1

for i in range(len(aData.FIC220)):
    if aData.FIC220[i] < 10:
       aData.FIC220[i] = '0'
      # print(job[i])
    else:
        pass
    i += 1

     
#########################################################
#속성별 추이
ord = 1
fig = plt.figure(figsize=(15, 100))
for i in range(len(aData.columns)):
    if i >= 5:
        j = i
        for k in range(4):
            ax = fig.add_subplot(25, 4, ord)
            plt.title(aData.columns[j])
            plt.plot(aData.iloc[0:, j:j+1])
    #ax.xaxis.set_visible(False)
    #ax.yaxis.set_visible(False)
    # x and y axis should be equal length
            x0,x1 = ax.get_xlim()
            y0,y1 = ax.get_ylim()
            ax.set_aspect(abs(x1-x0)/abs(y1-y0))
            j += 21
            ord += 1
    else:
        pass

plt.show()
########################################################

# column 추가
aData["rateNOx"] = (aData['NOX']/aData['STEAM'])*100
     
aData.corr()
targetData.corr()
corrDF = aData.corr()
corrDF['NOX']
corrDF['NOX'].sort_values(ascending=False)

corrDF['rateNOx'].sort_values(ascending=False)

#################################################
betaData = aData.where(aData['rateNOx']<40)
betaData = betaData.dropna(axis=0)
betaData.isnull().sum()
betaData.info()

feature_cols = ['FIC220', 'FIC201', 'FIC221']
X = betaData[feature_cols]
y = betaData.rateNOx


plt.plot(aData['rateNOx'])
plt.plot(betaData['rateNOx'])
# follow the usual sklearn pattern: import, instantiate, fit

lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)
zip(feature_cols, lm.coef_)
lm.score(X, y)



sortData = aData.sort_values('rateNOx', inplace=False)
sortData = sortData.reset_index()
sortData.head()

###################################################
plt.figure(figsize=(20,10))
plt.plot(sortData['FIC220'], label = 'FIC220')
plt.plot(sortData['FIC201'], label = 'FIC201')
plt.plot(sortData['FIC221'], label = 'FIC221')
plt.plot(sortData['FIC220']+sortData['FIC201']+sortData['FIC221'])
plt.legend()
plt.show()

plt.subplot(1, 3, 1)
plt.title('FIC220')
sortData['FIC220'].hist(bins=70)
plt.subplot(1, 3, 2)
sortData['FIC201'].hist(bins=70)
plt.subplot(1, 3, 3)
sortData['FIC221'].hist(bins=70)

plt.plot(sortData['STEAM'], label = 'STEAM')
plt.plot(sortData['NOX'], label = 'NOX')
plt.plot(sortData['rateNOx'], label = 'rateNOx')
plt.legend()
plt.show()

sortData.describe()
sortData.info()


corrDF = sortData.corr()
corrDF['rateNOx'].sort_values(ascending=False)
########################################################
#corr
colormap = plt.cm.viridis
plt.figure(figsize=(10 ,10))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(targetData.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

drop_elements = ['index', 'TI697', 'FI694', 'PI697',
       'PI699', 'TI698', 'FQ695', 'PC698', 'PDI691', 'FI698', 'TI720', 'PI693',
       'TI693', 'PI694', 'FI693', 'TI694', 'PI695', 'FC691', 'PI691', 'TI691',
       'LC691', 'PI692', 'TI797', 'FI794', 'PI797', 'PI799', 'TI798', 'FQ795',
       'PC798', 'PDI791', 'FI798', 'TI820', 'PI793', 'TI793', 'PI794', 'FI793',
       'TI794', 'PI795', 'FC791', 'PI791', 'TI791', 'LC791', 'PI792', 'TI897',
       'FI894', 'PI897', 'PI899', 'TI898', 'FQ895', 'PC898', 'PDI891', 'FI898',
       'TI920', 'PI893', 'TI893', 'PI894', 'FI893', 'TI894', 'PI895', 'FC891',
       'PI891', 'TI891', 'LC891', 'PI892', 'TI910', 'FC994', 'PI997', 'PI999',
       'TI998', 'FC995', 'PI998', 'PDC991', 'PI993', 'TI993', 'PI994', 'FI993',
       'TI994', 'PI995', 'FC991', 'PI991', 'TI991', 'LC991', 'PI990']
targetData = sortData.drop(drop_elements, axis = 1)


sumOfGas = sortData['FIC220']+sortData['FIC201']+sortData['FIC221']
plt.scatter(sortData['STEAM'],sumOfGas)
plt.scatter(sortData['rateNOx'],sumOfGas)
plt.scatter(sortData['NOX'],sumOfGas)
plt.scatter(sortData['FIC220'],sortData['rateNOx'])
plt.scatter(sortData['FIC201'],sortData['rateNOx'])
plt.scatter(sortData['FIC221'],sortData['rateNOx'])