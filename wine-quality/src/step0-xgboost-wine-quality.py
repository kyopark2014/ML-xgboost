#!/usr/bin/env python
# coding: utf-8

# # XGBoost - Wine Quality (Regression)
# 
# [UCI - Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality)

# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import numpy as np
import pandas as pd
import time


# In[4]:


df = pd.read_csv('./data/wine_concat.csv')


# In[5]:


df.head()


# In[6]:


df.isnull().sum().sum()


# In[7]:


df['quality'].value_counts()


# ### Splint Feature/Target Dataset

# In[8]:


y = df['quality']
X = df.drop(['quality'], axis=1)


# In[9]:


X = pd.get_dummies(X)  
X.head()


# ## Regression Model Selection

# In[10]:


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


# In[11]:


cross_validation(XGBRegressor(booster='gbtree'))


# In[12]:


cross_validation(XGBRegressor(booster='gblinear'))


# In[13]:


cross_validation(XGBRegressor(booster='dart', one_drop=1))


# In[14]:


from sklearn.linear_model import LinearRegression, LogisticRegression

cross_validation(LinearRegression())


# In[15]:


from sklearn.linear_model import Lasso

cross_validation(Lasso())


# In[16]:


from sklearn.linear_model import Ridge

cross_validation(Ridge())


# In[17]:


from sklearn.ensemble import RandomForestRegressor

cross_validation(RandomForestRegressor())


# ## HPO

# ### Default

# In[18]:


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split train/test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

# Utility function to estimate accuracy scores
def model_estimation(model):    
    y_pred = model.predict(X_test)

    reg_mse = mean_squared_error(y_test, y_pred)
    reg_rmse = np.sqrt(reg_mse)

    print('RMSE: %0.3f' % (reg_rmse))   
    
    return reg_rmse


# In[19]:


# n_estimators
param = 'n_estimators'
values = [50, 100, 200, 400, 800]
best_param = 0
best_score = 1e9
cnt = 0
start = time.time()

for value in values:
    start = time.time()
    
    xgb = XGBRegressor(booster='gbtree', n_estimators=value, 
                       random_state=2, verbosity=0, use_label_encoder=False, n_jobs=-1)
    
    xgb.fit(X_train, y_train)
    
    y_pred = xgb.predict(X_test)
    
    reg_mse = mean_squared_error(y_test, y_pred)
    reg_rmse = np.sqrt(reg_mse)
    
    if best_score > reg_rmse:
        best_score = reg_rmse
        best_param = cnt

    print('RMSE: %0.3f' % (reg_rmse))
    cnt = cnt + 1

print('Elapsed time: %0.2fs' % (time.time()-start))            
print('\nElased time: %0.2fs' % (time.time()-start))    
print('best score: %0.2f' % (best_score))
print('best param: ', values[best_param])


# In[20]:


# basic model
model = XGBRegressor(booster='gbtree', random_state=2, verbosity=0, use_label_encoder=False, n_jobs=-1)

model.fit(X_train, y_train)    

best_rmse = model_estimation(model)
best_model = model

print('\nbest_score: %0.3f' % (best_rmse))


# In[21]:


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


# ## HPO: Bayesian Optimization 

# In[22]:


from bayes_opt import BayesianOptimization

def xgbc_cv(n_estimators, learning_rate, max_depth, gamma, min_child_weight, subsample, colsample_bytree, ):
    
    
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

hyperparameter_space = {
    'n_estimators': (50, 800),
    'learning_rate': (0.01, 1.0),
    'max_depth': (1, 8),
    'gamma' : (0.01, 1),
    'min_child_weight': (1, 20),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.1, 1)
}

start = time.time()

optimizer = BayesianOptimization(f=xgbc_cv, pbounds=hyperparameter_space, random_state=2, verbose=0)
optimizer.maximize(init_points=5, n_iter=20, acq='ei')

print('Elapsed time: %0.2fs' % (time.time()-start)) 
optimizer.max


# In[23]:


best_params = optimizer.max['params']


# In[24]:


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

if(reg_rmse < best_rmse): 
    best_rmse = reg_rmse
    best_model = model
    
    print('\nbest_score: %0.3f' % (best_rmse))                     


# ## Evaluation & Save

# In[25]:


model_estimation(best_model)    


# In[26]:


print(best_model.get_params())


# In[28]:


model_name = "./output/xgboost_wine_quality.json"

model.save_model(model_name)


# ## Feature Importances

# In[29]:


print('Feature Importances:')
print(best_model.feature_importances_)

import xgboost as xgb

feature_data = xgb.DMatrix(X_test)
best_model.get_booster().feature_names = feature_data.feature_names
best_model.get_booster().feature_types = feature_data.feature_types

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 8))
xgb.plot_importance(best_model, ax=ax, importance_type='gain')

get_ipython().system('pip install graphviz')

xgb.plot_tree(best_model, num_trees=0, rankdir='LR')

fig = plt.gcf()
fig.set_size_inches(20, 10)
plt.show()

