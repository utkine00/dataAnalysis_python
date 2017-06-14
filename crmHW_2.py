# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:23:55 2017

@author: Seo, Jeong yeon
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#1. Read File
###.Please input path for the 'dataset.tsv' ###
input_file = 'D:\dataAnalysis_python\dataset.tsv'

crmData = pd.read_table(input_file, encoding='CP949')
crmData.head()
crmData.describe()
crmData.info()


#2. Data preprocessing
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


#2-1. 결측치 제거
crmData = crmData.dropna(axis=0)
crmData.isnull().sum()

#2-2. input, target 변수로 분리
### "x10", "x11"는 마지막으로 고객에게 연락하 날로 연도 정보 및 현재 시점을 알 수 없어 최근일을 판단 할 수 없다고
### 생각되어 분석에서 제외하였습니다.
X_input = crmData.drop("y", axis=1)
X_input = X_input.drop("x10", axis=1)
X_input = X_input.drop("x11", axis=1)
Y_target = crmData["y"]

# train, test set으로 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(X_input, Y_target, test_size=0.2, random_state=0)
x_train.shape, y_train.shape
x_test.shape, y_test.shape

#3. 예측 모델: Random forest 알고리듬으로 적용
### 적용 이유: Input data가 숫자형 및 범주형으로 구성되어 있을 때 이를 동시에 다룰수 있는 방법으로 ###
### decision tree의 overfitting 문제를 방지할 수 있는 알고리듬이기 때문에 적용해보았습니다.       ###
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)

random_forest.score(x_train, y_train)
random_forest.score(x_test, y_test)

pscore = metrics.accuracy_score(y_test, y_pred)

print('\n********************Predition accuracy for the test data  = ', pscore, '********************')

submission = pd.DataFrame({
        "prediction": y_pred,
        "original": y_test,
        "x1": x_test["x1"],
        "x2": x_test["x2"],
        "x3": x_test["x3"],
        "x4": x_test["x4"],
        "x5": x_test["x5"],
        "x6": x_test["x6"],
        "x7": x_test["x7"],
        "x8": x_test["x8"],
        "x9": x_test["x9"],
        "x12": x_test["x12"],
        "x13": x_test["x13"],
        "x14": x_test["x14"],
        "x15": x_test["x15"],             
                                
    })
submission.to_csv('comparison.csv', index=True)
y_test.value_counts(normalize = True)



### 분류기 TEST
#testData = [x1,   x2,   x3,   x4,   x5,   x6,   x7,   x8,   x9,    x12,  x13,    x14,    x15 ]
testData = [36,   0.0,   0,   1,   0,  1224,   1,   0,   0,    1,  374,    1,    0 ]
Y_pred = random_forest.predict(testData)
print('\n****************Predition value  = ', Y_pred, '********************')




###############################################################################
#데이터 분포 탐색
#crmData['y'].value_counts(normalize = True)
#crmData['y'][crmData['x3'] == 'married'].value_counts(normalize=True)
#crmData['y'][crmData['x3'] == 'single'].value_counts(normalize=True)
#crmData['y'][crmData['x6'] < 1000].value_counts(normalize=True)
#crmData['y'][crmData['x1'] > 60 ].value_counts(normalize=True)
#crmData['y'][crmData['x5'] == 'yes' ].value_counts(normalize=True)
#crmData['y'][crmData['x5'] == 'no' ].value_counts(normalize=True)
#crmData['y'][crmData['x15'] == 0 ].value_counts(normalize=True)
#crmData['x15'][crmData['y'] == 1 ].value_counts(normalize=True)

#plt.scatter(crmData['x14'], crmData['y'], s=40)
#plt.scatter(crmData['x15'], crmData['y'], s=40)

title_xt = pd.crosstab(crmData['x2'], crmData['y'])
title_xt_pct = title_xt.div(title_xt.sum(1).astype(float), axis=0)

title_xt_pct.plot(kind='bar', 
                  stacked=True, 
                  title='Fixed deposit')
plt.xlabel('job')
plt.ylabel('rate')


###############################################################################



