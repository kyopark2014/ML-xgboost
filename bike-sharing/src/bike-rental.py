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

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

# load dataset 
print('start:')
df_bikes = pd.read_csv('../data/bike_rentals_cleaned.csv')

# define feature and target
X = df_bikes.iloc[:,:-1]
y = df_bikes.iloc[:,-1]

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score

#kfold = KFold(n_splits=5, shuffle=True, random_state=2)
#kfold = StratifiedKFold(n_splits=5)

def cross_validation(model):
    start = time.time()
    
    #scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kfold)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)

    rmse = np.sqrt(-scores)
    
    print('Cross Validation:')
    print('Elased time: %0.2fs' % (time.time()-start))
    print('RMSE:', np.round(rmse, 3))
    print('Avg. RMSE: %0.3f' % (rmse.mean()))

cross_validation(XGBRegressor(booster='gbtree'))

# Split train/test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score(=RMSE): {0:.3f}".format(
                    np.sqrt(-results["mean_test_score"][candidate]),
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")

def randomized_search(params, runs=20): 
    xgb = XGBRegressor(booster='gbtree', random_state=2, verbosity=0, use_label_encoder=False, n_jobs=-1)
    
    # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)    
    # rand_reg = RandomizedSearchCV(xgb, params, cv=kfold, n_iter=runs, n_jobs=-1, random_state=2, scoring='neg_mean_squared_error')    
    rand_reg = RandomizedSearchCV(xgb, params, cv=10, n_iter=runs, n_jobs=-1, random_state=2, scoring='neg_mean_squared_error')
    
    rand_reg.fit(X_train, y_train)    
    
    best_model = rand_reg.best_estimator_    
    
    best_params = rand_reg.best_params_
    print("best parameter:", best_params)
    
    best_score = rand_reg.best_score_
    rmse = np.sqrt(-best_score)
    print("best score: {:.3f}".format(rmse))

    print("rand_reg.cv_results_: ", rand_reg.cv_results_)

    report(rand_reg.cv_results_)
    
    return best_model

start = time.time()

best_model = randomized_search(
    params={
        'n_estimators':[50,100,800],
        'learning_rate':[0.1],
        'max_depth':[2],
        'subsample':[0.9],
        }, 
    runs=15)



print('\nElapsed time: %0.2fs' % (time.time()-start))

#best_model.save_model(model_location)
