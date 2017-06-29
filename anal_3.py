# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 08:56:04 2017

@author: admin
"""

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

#1. Read File
###.Please input path for the 'dataset.csv' ###
input_file = 'D:\pyAnal\MODELING_BASE.csv'
aData = pd.read_table(input_file, encoding='CP949', delimiter=',')

#2. Data information
aData.head()
aData.describe()
aData.info()
#aData.values
aData.columns
aData.isnull().sum()
aData = aData.dropna(axis=0)

#3. data preprocessing

for i in range(len(aData.FIC201)):
    if aData.FIC201[i] < 10:
       aData.FIC201[i] = '0'
    else:
        pass
    i += 1

for i in range(len(aData.FIC220)):
    if aData.FIC220[i] < 10:
       aData.FIC220[i] = '0'
    else:
        pass
    i += 1

for i in range(len(aData.FIC221)):
    if aData.FIC221[i] < 10:
       aData.FIC221[i] = '0'
    else:
        pass
    i += 1

    
aData["rateNOx"] = (aData['NOX']/aData['STEAM'])*100

betaData = aData.where( aData["rateNOx"] < 60 )
betaData = betaData.dropna(axis=0)
betaData = betaData.where( betaData["rateNOx"] > 30 )
betaData = betaData.dropna(axis=0)
betaData.info()
betaData.describe()

betaData.loc[ betaData['rateNOx'] <= 41.109631, 'lvlOfNOx'] = 0
betaData.loc[(betaData['rateNOx'] > 41.109631) & (betaData['rateNOx'] <= 47.148855), 'lvlOfNOx'] = 1
betaData.loc[(betaData['rateNOx'] > 47.148855) & (betaData['rateNOx'] <= 49.659183), 'lvlOfNOx'] = 2
betaData.loc[betaData['rateNOx'] > 49.659183, 'lvlOfNOx'] = 3

betaData['lvlOfNOx'].hist(bins=70)


group0 = betaData.where( betaData['lvlOfNOx'] == 0 )
group0 = group0.dropna(axis=0)
group0 = group0.reset_index()

group1 = betaData.where( betaData['lvlOfNOx'] == 1 )
group1 = group1.dropna(axis=0)
group1 = group1.reset_index()

group2 = betaData.where( betaData['lvlOfNOx'] == 2 )
group2 = group2.dropna(axis=0)
group2 = group2.reset_index()

group3 = betaData.where( betaData['lvlOfNOx'] == 3 )
group3 = group3.dropna(axis=0)
group3 = group3.reset_index()

plt.subplot(2, 2, 1)
plt.plot(group0['FIC220'], label = 'FIC220')
plt.plot(group0['FIC201'], label = 'FIC201')
plt.plot(group0['FIC221'], label = 'FIC221')
plt.subplot(2, 2, 2)
plt.plot(group1['FIC220'], label = 'FIC220')
plt.plot(group1['FIC201'], label = 'FIC201')
plt.plot(group1['FIC221'], label = 'FIC221')
plt.subplot(2, 2, 3)
plt.plot(group2['FIC220'], label = 'FIC220')
plt.plot(group2['FIC201'], label = 'FIC201')
plt.plot(group2['FIC221'], label = 'FIC221');
plt.subplot(2, 2, 4)
plt.plot(group3['FIC220'], label = 'FIC220')
plt.plot(group3['FIC201'], label = 'FIC201')
plt.plot(group3['FIC221'], label = 'FIC221')
plt.legend()
plt.show()

lvl1 = betaData.where( betaData['rateNOx'] <= 41.109631 )
lvl1 = lvl1.dropna(axis=0)
lvl1 = lvl1.reset_index()

lvl2 = betaData.where( betaData['rateNOx'] > 41.109631 )
lvl2 = lvl2.dropna(axis=0)
lvl2 = lvl2.reset_index()

plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
plt.plot(lvl1['FIC220'], label = 'FIC220')
plt.plot(lvl1['FIC201'], label = 'FIC201')
plt.plot(lvl1['FIC221'], label = 'FIC221')
plt.subplot(1, 2, 2)
plt.plot(lvl2['FIC220'], label = 'FIC220')
plt.plot(lvl2['FIC201'], label = 'FIC201')
plt.plot(lvl2['FIC221'], label = 'FIC221')
plt.legend()
plt.show()

lvl1.describe()
lvl2.describe()



group201_2 = betaData.where( betaData['FIC201'] >= 458.995622 )
group201_2 = group201_2.dropna(axis=0)
group201_2 = group201_2.reset_index()

group201 = betaData.where( betaData['FIC201'] < 458.995622 )
group201 = group201.dropna(axis=0)
group201 = group201.reset_index()

group201.describe()
group201_2.describe()

plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
plt.plot(group201['FIC220'], label = 'FIC220')
plt.plot(group201['FIC201'], label = 'FIC201')
plt.plot(group201['FIC221'], label = 'FIC221')
plt.subplot(1, 2, 2)
plt.plot(group201_2['FIC220'], label = 'FIC220')
plt.plot(group201_2['FIC201'], label = 'FIC201')
plt.plot(group201_2['FIC221'], label = 'FIC221')
plt.legend()
plt.show()


plt.plot(group201_2['NOX'], label = 'Is201')
plt.plot(group201['NOX'], label = 'Not201')
plt.legend()
plt.show()
plt.plot(group201['NOX'])   

plt.plot(group201['rateNOx'], label = 'FIC201')
plt.plot(group201_2['rateNOx'], label = 'FIC201_2')
plt.legend()
plt.show()

plt.plot(group201['FIC220'], label = 'FIC220')
plt.plot(group201['FIC201'], label = 'FIC201')
plt.plot(group201['FIC221'], label = 'FIC221')
#plt.plot(aData['FIC220']+aData['FIC201']+aData['FIC221'])
plt.legend()
plt.show()


steamSortData = betaData.sort_values('STEAM', inplace=False)
steamSortData = steamSortData.reset_index()
steamSortData.head()
plt.figure(figsize=(20,10))
plt.plot(steamSortData['STEAM'])
plt.plot(steamSortData['NOX'])   
plt.plot(steamSortData['rateNOx'])
plt.plot(steamSortData['FIC220'], label = 'FIC220')
plt.plot(steamSortData['FIC201'], label = 'FIC201')
plt.plot(steamSortData['FIC221'], label = 'FIC221')
plt.figure(figsize=(20,10))
plt.plot(steamSortData['FIC220']+steamSortData['FIC201']+steamSortData['FIC221'])
plt.legend()
plt.show()








feature_cols = ['FIC220', 'FIC201', 'FIC221']
X = lvl1[feature_cols]
y = lvl1.rateNOx

plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
plt.plot(lvl1['rateNOx'], label = 'lvl1')
plt.subplot(1, 2, 2)
plt.plot(lvl2['rateNOx'], label = 'lvl2')


plt.plot(aData['rateNOx'])
# follow the usual sklearn pattern: import, instantiate, fit








plt.plot(betaData['STEAM'])
plt.plot(betaData['NOX'])   
plt.plot(betaData['NOX']/betaData['STEAM'])
plt.plot(betaData['FIC220'], label = 'FIC220')
plt.plot(betaData['FIC201'], label = 'FIC201')
plt.plot(betaData['FIC221'], label = 'FIC221')
#plt.plot(aData['FIC220']+aData['FIC201']+aData['FIC221'])
plt.legend()
plt.show()

plt.plot(aData['STEAM'])
plt.plot(aData['NOX'])
plt.plot(aData['NOX']/aData['STEAM'])
plt.plot(aData['FIC220'], label = 'FIC220')
plt.plot(aData['FIC201'], label = 'FIC201')
plt.plot(aData['FIC221'], label = 'FIC221')
#plt.plot(aData['FIC220']+aData['FIC201']+aData['FIC221'])
plt.legend()
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