#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:55:12 2022

@author: parkjh
"""

#%% library load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import geopy.distance
from sklearn.ensemble import RandomForestRegressor

path = "./data/"

# data load and preprocessing
train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path+"test.csv")
bts = pd.read_csv(path+"bus_bts.csv")

def g(x):
    if x in ['2019-09-12','2019-09-13','2019-09-14','2019-10-03','2019-10-09']:
        return 1
    else:
        return 0
train['holiday'] = train['date'].apply(g)
test['holiday'] = test['date'].apply(g)

# datetime weekday one hot encoding
train['date'] = pd.to_datetime(train['date'])
train['weekday'] = train['date'].dt.weekday
train = pd.get_dummies(train,columns=['weekday'])

test['date'] = pd.to_datetime(test['date'])
test['weekday'] = test['date'].dt.weekday
test = pd.get_dummies(test,columns=['weekday'])

# 시내, 시외버스 구분
train['in_out'] = train['in_out'].map({'시내':0,'시외':1})
test['in_out'] = test['in_out'].map({'시내':0,'시외':1})

# 제주, 서귀포시와의 거리
coords_jeju = (33.500770, 126.522761)
coords_seogwipo = (33.259429, 126.558217)

train['dis_jeju'] = [geopy.distance.geodesic((train['latitude'].iloc[i],train['longitude'].iloc[i]), coords_jeju).km for i in range(len(train))]
train['dis_seogwipo'] = [geopy.distance.geodesic((train['latitude'].iloc[i],train['longitude'].iloc[i]), coords_seogwipo).km for i in range(len(train))]

test['dis_jeju'] = [geopy.distance.geodesic((test['latitude'].iloc[i],test['longitude'].iloc[i]), coords_jeju).km for i in range(len(test))]
test['dis_seogwipo'] = [geopy.distance.geodesic((test['latitude'].iloc[i],test['longitude'].iloc[i]), coords_seogwipo).km for i in range(len(test))]

#
rain = pd.read_csv(path + "rain_jeju_9&10.csv")
rain['date'] = pd.to_datetime(rain['date'])
rain['date'] = rain["date"].dt.strftime("%Y-%m-%d")
rain['date'] = pd.to_datetime(rain['date'])
rain.head()
train = pd.merge(train,rain)
test = pd.merge(test,rain)

#%% 독립변수, 종속변수 정의
input_var=['in_out','6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride',
       '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff', '8~9_takeoff',
       '9~10_takeoff', '10~11_takeoff', '11~12_takeoff', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6', 'dis_jeju', 'dis_seogwipo', 'holiday', 'rain_jeju(mm)', 'rain_seogwipo(mm)']
target=['18~20_ride']

# RMSE 손실함수 정의

from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

def rmse(y, y_pred):
    return mean_squared_error(y, y_pred)**0.5

rmse_scorer = make_scorer(rmse, greater_is_better=False)

#%% RF 모델 정의, 학습, 평가
X_train=train[input_var]
y_train=train[target]
X_test=test[input_var]

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=140, random_state=1217)

rf.fit(X_train,y_train)
scores = cross_val_score(rf, X_train, y_train, scoring=rmse_scorer, cv=5)
print(-scores)

#full_var: array([-2.93831871, -2.93287175, -3.37310352, -2.98796676, -2.6773062 ])

#%% RF 모델 튜닝
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rf_params = {'n_estimators' : [140, 160, 180, 200, 220]}
rf = RandomForestRegressor(random_state=1217)

gridsearch_forest = GridSearchCV(rf, rf_params, scoring=rmse_scorer, cv=5, n_jobs=-1)

gridsearch_forest.fit(X_train, y_train)
print(gridsearch_forest.best_params_)


#%% 출력
test['18~20_ride'] = rf.predict(X_test)
test[['id','18~20_ride']].to_csv("subm_RF1217.csv",index=False)























