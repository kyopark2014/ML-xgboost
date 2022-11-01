import numpy as np
import pandas as pd
import time

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

from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Split the dataframe into test and train data

def split_data(df):
    y = df['quality']
    X = df.drop(['quality'], axis=1)

    X = pd.get_dummies(X)  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2) # Split train/test dataset

    return X_train, X_test, y_train, y_test

# HPO: Bayesian Optimization
def xgbc_cv(X_train, X_test, y_train, y_test, n_estimators, learning_rate, max_depth, gamma, min_child_weight, subsample, colsample_bytree):
    min_rmse = 1e10

    xgb = XGBRegressor(booster='gbtree', objective='reg:squarederror',
                        n_estimators=int(n_estimators),
                        learning_rate=learning_rate,
                        max_depth=int(max_depth),
                        gamma=gamma,
                        min_child_weight=min_child_weight,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        random_state=2, verbosity=0, use_label_encoder=False, n_jobs=-1)

    xgb.fit(X_train, y_train)    

    y_pred = xgb.predict(X_test)

    reg_mse = mean_squared_error(y_test, y_pred)
    reg_rmse = np.sqrt(reg_mse)

    print('RMSE: %0.3f' % (reg_rmse))   
    
    return -reg_rmse

# HPO
def optimize_hyperparamter(X_train, X_test, y_train, y_test, hyperparameter_space):
    start = time.time()
     
    optFunc = lambda n_estimators, learning_rate, max_depth, gamma, min_child_weight, subsample, colsample_bytree: xgbc_cv(X_train, X_test, y_train, y_test, n_estimators, learning_rate, max_depth, gamma, min_child_weight, subsample, colsample_bytree)

    optimizer = BayesianOptimization(f=optFunc, pbounds=hyperparameter_space, random_state=2, verbose=0)

    optimizer.maximize(init_points=5, n_iter=20, acq='ei')

    print('Elapsed time: %0.2fs' % (time.time()-start))   
    print(optimizer.max)

    # evaluation
    best_params = optimizer.max['params']
    print(best_params)

    return best_params

# get the best model
def get_best_model(X_train, X_test, y_train, y_test, best_params):
    start = time.time()
    model = XGBRegressor(booster='gbtree', objective='reg:squarederror',
                            n_estimators=int(best_params['n_estimators']), 
                            learning_rate=best_params['learning_rate'], 
                            max_depth=int(best_params['max_depth']), 
                            gamma=best_params['gamma'], 
                            min_child_weight=int(best_params['min_child_weight']), 
                            subsample=best_params['subsample'], 
                            colsample_bytree=best_params['colsample_bytree'],
                            random_state=2, verbosity=0, use_label_encoder=False, n_jobs=-1)

    model.fit(X_train, y_train)    

    y_pred = model.predict(X_test)

    reg_mse = mean_squared_error(y_test, y_pred)
    reg_rmse = np.sqrt(reg_mse)

    print('Elapsed time: %0.2fs' % (time.time()-start))        
    print('RMSE: %0.3f' % (reg_rmse))  

    return model

def main():
    # Load Data
    df = pd.read_csv('../data/wine_concat.csv')

    X_train, X_test, y_train, y_test = split_data(df)

    hyperparameter_space = {
        'n_estimators': (50, 800),
        'learning_rate': (0.01, 1.0),
        'max_depth': (1, 8),
        'gamma' : (0.01, 1),
        'min_child_weight': (1, 20),
        'subsample': (0.5, 1),
        'colsample_bytree': (0.1, 1)
    }

    best_params = optimize_hyperparamter(X_train, X_test, y_train, y_test, hyperparameter_space)

    best_model = get_best_model(X_train, X_test, y_train, y_test, best_params)

    model_name = "../output/xgboost_wine_quality.json"
    best_model.save_model(model_name)

main()
