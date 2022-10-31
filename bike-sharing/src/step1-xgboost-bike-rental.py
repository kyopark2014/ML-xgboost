import os
import sys
import pickle
import xgboost as xgb
import argparse
import json
import numpy as np
import pandas as pd
import time
import torch


# load dataset 
df_bikes = pd.read_csv('../data/bike_rentals_cleaned.csv')

# define feature and target
X = df_bikes.iloc[:,:-1]
y = df_bikes.iloc[:,-1]

from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score

def cross_validation(model):
    start = time.time()
    
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)

    rmse = np.sqrt(-scores)
    
    print('Cross Validation:')
    print('Elased time: %0.2fs' % (time.time()-start))
    print('RMSE:', np.round(rmse, 3))
    print('Avg. RMSE: %0.3f' % (rmse.mean()))

cross_validation(XGBRegressor(booster='gbtree'))