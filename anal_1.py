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

plt.plot(aData['NOX']/aData['STEAM'])
plt.plot((aData['STEAM']-aData['NOX'])/aData['STEAM'])

plt.plot(aData['FIC220'])
plt.plot(aData['FIC201'])
plt.plot(aData['FIC221'])


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