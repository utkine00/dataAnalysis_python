# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:34:38 2017

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


#delimiter='\t'

#2. Data information
aData.head()
aData.describe()
aData.info()
aData.values
aData.columns
aData.isnull().sum()


aData['STEAM']

plt.figure(figsize=(28,140))
plt.scatter(aData['STEAM'], aData['NOX'])
plt.plot(aData['STEAM'])
plt.plot(aData['NOX'])
#rateNOx = (aData['NOX']/aData['STEAM'])*100

plt.plot(aData['NOX']/aData['STEAM'])
plt.plot((aData['STEAM']-aData['NOX'])/aData['STEAM'])

plt.plot(aData['FIC220'], label = 'FIC220')
plt.plot(aData['FIC201'], label = 'FIC201')
plt.plot(aData['FIC221'], label = 'FIC221')
plt.plot(aData['FIC220']+aData['FIC201']+aData['FIC221'])
plt.legend()
plt.show()

aData.corr()
targetData.corr()
corrDF = aData.corr()
corrDF['NOX']
corrDF['NOX'].sort_values(ascending=False)

aData["rateNOx"] = (aData['NOX']/aData['STEAM'])*100
corrDF['rateNOx'].sort_values(ascending=False)



sumOfGas = aData['FIC220']+aData['FIC201']+ aData['FIC221']
plt.plot(sumOfGas)
plt.plot(aData['NOX']/aData['STEAM'])
plt.plot(aData['NOX']/(aData['STEAM']+aData['NOX']))
plt.plot(aData['STEAM'])
plt.plot(aData['NOX'])
plt.scatter(aData['NOX'],sumOfGas)

plt.plot(aData['FIC221']/sumOfGas, label = 'Rate(FIC221)')
plt.plot(aData['FIC201']/sumOfGas, label = 'Rate(FIC201)')
plt.plot(aData['FIC220']/sumOfGas, label = 'Rate(FIC220)')
plt.plot(aData['NOX']/aData['STEAM'], label = 'Rate(NOX)')
plt.legend()
plt.show()

plt.plot(aData['FI694'])
plt.plot(aData['FI794'])
plt.plot(aData['FI894'])
plt.plot(aData['FC994'])
sumfuel = aData['FI694']+ aData['FI794']+ aData['FI894']+aData['FC994']
plt.plot(sumfuel)

plt.plot(aData['FIC221']/sumOfGas, label = 'Rate(FIC221)')
plt.plot(aData['FIC201']/sumOfGas, label = 'Rate(FIC201)')
plt.plot(aData['FIC220']/sumOfGas, label = 'Rate(FIC220)')
plt.plot(aData['NOX']/aData['STEAM'], label = 'Rate(NOX)')
plt.plot(sumfuel, label = 'sumfuel')
plt.legend()
plt.show()


plt.plot(aData['FC691'])
plt.plot(aData['FC791'])
plt.plot(aData['FC891'])
plt.plot(aData['FC991'])
feedWT = aData['FC691']+ aData['FC791']+ aData['FC891']+aData['FC991']
plt.plot(feedWT)
plt.plot(aData['STEAM'])
plt.plot(aData['NOX'])
plt.plot(aData['STEAM']-feedWT)
plt.plot(aData['STEAM']+aData['NOX'], label = 'SUM')
plt.plot(feedWT, label = 'feedWT')
plt.legend()
plt.show()
plt.plot(aData['STEAM']/feedWT)


aData["steamWT"] = aData['STEAM']/feedWT


plt.plot(aData['PI691'])
plt.plot(aData['PI791'])
plt.plot(aData['PI891'])
plt.plot(aData['PI991'])
feedWTPre = aData['PI691']+ aData['PI791']+ aData['PI891']+aData['PI991']

plt.plot(aData['TI691'])
plt.plot(aData['TI791'])
plt.plot(aData['TI891'])
plt.plot(aData['TI991'])
feedWTTem = aData['TI691']+ aData['TI791']+ aData['TI891']+aData['TI991']

plt.plot(aData['FC691'])
plt.plot(aData['FC791'])
plt.plot(aData['FC891'])
plt.plot(aData['FC991'])
feedWT = aData['FC691']+ aData['FC791']+ aData['FC891']+aData['FC991']



plt.subplot(1, 3, 1)
plt.plot(aData['FIC220'])
plt.subplot(1, 3, 2)
plt.plot(aData['FIC201'])
plt.subplot(1, 3, 3)
plt.plot(aData['FIC221'])

plt.subplot(2, 2, 1)
plt.plot(aData['TI697'])
plt.subplot(2, 2, 2)
plt.plot(aData['TI797'])
plt.subplot(2, 2, 3)
plt.plot(aData['TI897'])
plt.subplot(2, 2, 4)
plt.plot(aData['TI910'])


plt.figure(figsize=(15,10))
plt.plot(aData['TI697'])
plt.plot(aData['TI797'])
plt.plot(aData['TI897'])
plt.plot(aData['TI910'])

aData.iloc[0:5, 1:2]


a = aData.columns[1]


len(aData.columns)


fig = plt.figure(figsize=(20, 100))

for i in range(len(aData.columns)):
    if i >= 2:
        ax = fig.add_subplot(25, 4, i-1)
        plt.plot(aData.iloc[0:, i:i+1])
    #ax.xaxis.set_visible(False)
    #ax.yaxis.set_visible(False)
    # x and y axis should be equal length
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect(abs(x1-x0)/abs(y1-y0))
    else:
        pass

plt.show()


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
# Feature selection
drop_elements = ['TI697', 'FI694', 'PI697',
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
targetData = aData.drop(drop_elements, axis = 1)

drop_elements = ['TI697', 'PI697',
       'PI699', 'TI698', 'FQ695', 'PC698', 'PDI691', 'FI698', 'TI720', 'PI693',
       'TI693', 'PI694', 'FI693', 'TI694', 'PI695', 'FC691', 'PI691', 'TI691',
       'LC691', 'PI692', 'TI797', 'PI797', 'PI799', 'TI798', 'FQ795',
       'PC798', 'PDI791', 'FI798', 'TI820', 'PI793', 'TI793', 'PI794', 'FI793',
       'TI794', 'PI795', 'FC791', 'PI791', 'TI791', 'LC791', 'PI792', 'TI897',
       'PI897', 'PI899', 'TI898', 'FQ895', 'PC898', 'PDI891', 'FI898',
       'TI920', 'PI893', 'TI893', 'PI894', 'FI893', 'TI894', 'PI895', 'FC891',
       'PI891', 'TI891', 'LC891', 'PI892', 'TI910', 'PI997', 'PI999',
       'TI998', 'FC995', 'PI998', 'PDC991', 'PI993', 'TI993', 'PI994', 'FI993',
       'TI994', 'PI995', 'FC991', 'PI991', 'TI991', 'LC991', 'PI990']
targetData = aData.drop(drop_elements, axis = 1)
########################################################
#corr
colormap = plt.cm.viridis
plt.figure(figsize=(10 ,10))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(targetData.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

########################################################
sns.set(style='whitegrid', context = 'notebook')
cols = ['STEAM', 'NOX', 'rateNOx']
sns.pairplot(aData[cols], size = 3, kind='reg')
plt.show()


########################################################

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

# create X and y
feature_cols = ['FIC220', 'FIC201', 'FIC221']
X = aData[feature_cols]
y = aData.rateNOx


plt.plot(aData['rateNOx'])
# follow the usual sklearn pattern: import, instantiate, fit

lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)
zip(feature_cols, lm.coef_)
lm.score(X, y)


aData = aData.dropna(axis=0)
aData.isnull().sum()

aData['rateNOx'].sort_values(ascending=False)

count(aData['rateNOx']>50)

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


########################################################


plt.figure()
plt.subplots_adjust(hspace=1.5, figsize=(30,20))
plt.subplots(figsize=(3,2))

fig, ax = plt.subplots(len(aData.columns), sharex=True, figsize=(30,100))
for i in range(len(aData.columns)):
    if i >= 2:
        ax[i].plot(aData.iloc[0:, i:i+1])
        #plt.subplot(25, 4, i+1)
        #plt.plot(aData.iloc[0:, i:i+1]) 
    else:
        pass
    i += 1

fig.subplots_adjust(hspace=0.3, wspace=0.05)    
fig.tight_layout()
for i in range(len(aData.columns)):
    if i >= 2:
        plt.subplot(25, 4, i+1)
        plt.plot(aData.iloc[0:, i:i+1]) 
    else:
        pass
    i += 1 
    


plt.subplots_adjust(left, bottom, right, top, wspace, hspace)



left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.5   # the amount of width reserved for blank space between subplots
hspace = 0.5   # the amount of height reserved for white space between subplots

    
    
    
fig, axes = plt.subplots(3, 6, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
fig.subplots_adjust(hspace=0.3, wspace=0.05)
for ax, interp_method in zip(axes.flat, methods):
    ax.imshow(X, interpolation=interp_method)
    ax.set_title(interp_method)
plt.show()   