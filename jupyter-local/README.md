# XGBoost - Local

[Froud Detection 예제](https://github.com/kyopark2014/ML-xgboost/blob/main/jupyter-local/xgboost-fraud-detection-pytorch.ipynb)에 대해 아래와 같이 설명합니다. 

## Hyperparameter

```python
max_depth = 3
eta = 0.2
objective = 'binary:logistic'
scale_pos_weight = 29
```

#### params 

- objective: reg:squarederror 또는 binary:logistic
- eta: 각 boosting step의 step size 
- max_depth: Tree의  maximum depth 
- nthread: Thread의 숫자

```python
hyperparameters = {
       "scale_pos_weight" : "29",    
        "max_depth": "3",
        "eta": "0.2",
        "objective": "binary:logistic",
        "num_round": "100",
}
```



## Cross Validation

아래와 같이 Cross Validation을 수행합니다. 

```python
cv_results = xgb.cv(
    params = params,
    dtrain = dtrain,
    num_boost_round = num_boost_round,
    nfold = nfold,
    early_stopping_rounds = early_stopping_rounds,
    metrics = ('auc'),
    stratified = True, # 레이블 (0,1) 의 분포에 따라 훈련 , 검증 세트 분리
    seed = 0)
```

이때 결과는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/52392004/190918913-c46b4a23-76ef-4ae2-ac8f-56ffee12e01a.png)

펑균값을 구하면 Train dataset은 0.94, Validation dataset은 0.82의 결과를 얻었습니다. 

```python
print(f"[0]#011train-auc:{cv_results.iloc[-1]['train-auc-mean']}")
print(f"[1]#011validation-auc:{cv_results.iloc[-1]['test-auc-mean']}")

[0]#011train-auc:0.9405190344815983
[1]#011validation-auc:0.8218406316371057
```

## Evaluation

Model과 Test dataset을 로드하고 데이터를 준비합니다. 

```python
model = xgboost.XGBRegressor()
model.load_model("xgboost-model-pytorch")

import pandas as pd
df = pd.read_csv('../dataset/test.csv')

y_test = df.iloc[:, 0].astype('int')    
df.drop(df.columns[0], axis=1, inplace=True)

X_test = df.values
```

predict()을 수행합니다. 

```python
predictions_prob = model.predict(X_test)

threshold = 0.5
predictions = [1 if e >= 0.5 else 0 for e in predictions_prob ] 
```

이



## Reference

[XGBoost - Cross Validation](https://rdrr.io/cran/xgboost/man/xgb.cv.html)

