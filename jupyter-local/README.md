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

[Evaluation 상세코드](https://github.com/kyopark2014/ML-xgboost/blob/main/jupyter-local/xgboost-evaluation-pytorch.ipynb)의 내용을 아래와 같이 설명합니다. 

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

이때, predictions_prob, predictions은 아래와 같은 값들을 가집니다. 

```python
print(predictions_prob[0:20])

[0.10092484 0.08251919 0.4293206  0.23539546 0.6692123  0.290863
 0.69971937 0.66698503 0.8422588  0.13136372 0.46003887 0.6096221
 0.01886459 0.11602865 0.7805386  0.14637303 0.31743652 0.1873799
 0.0977393  0.11276104]

print(predictions[0:20])
[0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
```

아래와 같이 결과를 정리할 수 있습니다. 

```python
from sklearn.metrics import classification_report

print(classification_report(y_true=y_test, y_pred = predictions))
```

이때의 결과는 아래와 같습니다. 세부항목의 의미는 [Confusion Matrix (오차행렬)](https://github.com/kyopark2014/ML-Algorithms/blob/main/confusion-matrix.md)에서 확인할 수 있습니다. 

```python
              precision    recall  f1-score   support

           0       0.99      0.72      0.83       967
           1       0.09      0.79      0.16        33

    accuracy                           0.72      1000
   macro avg       0.54      0.75      0.50      1000
weighted avg       0.96      0.72      0.81      1000
```

confusion_matrix를 구하면 아래와 같습니다. 

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true= y_test, y_pred= predictions)    
print(cm)
```

이때의 결과는 아래와 같습니다. 

```python
[[697 270]
 [  7  26]]
```

아래와 같이 MSE와 표준편차를 구할 수 있습니다. 
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions)
print(mse)

import numpy as np
std = np.std(y_test - predictions)
print(std)
```

이때의 값은 MSE가 0.277이고, 표준편차가 0.4558848538830812 입니다. 



## Reference

[XGBoost - Cross Validation](https://rdrr.io/cran/xgboost/man/xgb.cv.html)

