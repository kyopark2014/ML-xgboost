import os
import sys
import pickle
import xgboost as xgb
import argparse
import pandas as pd
import json

import pandas as pd
pd.options.display.max_rows=20
pd.options.display.max_columns=10


# load dataset 
df_bikes = pd.read_csv('./data/bike_rentals_cleaned.csv')

# define feature and target
X = df_bikes.iloc[:,:-1]
y = df_bikes.iloc[:,-1]

from sklearn.model_selection import RandomizedSearchCV

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
    
    return best_model

start = time.time()

best_model = randomized_search(
    params={
        'n_estimators':[800],
        'learning_rate':[0.1],
        'max_depth':[2],
        'subsample':[0.9],
        }, 
    runs=20)

print('\nElapsed time: %0.2fs' % (time.time()-start))

best_model.save_model(model_location)
