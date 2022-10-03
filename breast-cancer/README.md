#  XGBoost로 breast cancer 분석하기 

## 데이터 가져오기 

[Breast_cancer_data.csv](https://github.com/Suji04/ML_from_Scratch/blob/master/Breast_cancer_data.csv)에서 아래처럼 데이터를 직접 가져올 수 있습니다. 여기의 데이터는 [Dataset - Breast Cancer Prediction with XGBoost](https://www.kaggle.com/code/armagansarikey/breast-cancer-prediction-with-xgboost/data)와 동일합니다. 

```python
datapath = "https://raw.githubusercontent.com/Suji04/ML_from_Scratch/master/Breast_cancer_data.csv"

import pandas as pd

data = pd.read_csv(datapath)
data.head()
```

이때의 결과는 아래와 같습니다. 

![image](https://user-images.githubusercontent.com/52392004/193499786-63c3c796-a176-48d6-8065-570e3a064930.png)

## XGBoost로 분석하기 

### Breast Cancer Prediction with XGBoost

[Breast Cancer Prediction with XGBoost](https://www.kaggle.com/code/armagansarikey/breast-cancer-prediction-with-xgboost)의 내용을 정리한 [xgboost-breast-canser.ipynb](https://github.com/kyopark2014/ML-xgboost/blob/main/breast-cancer/xgboost-breast-canser.ipynb)에 대해 설명합니다. 

아래와 같이 XGBClassifier를 정의하고 학습을 수행합니다. 

```python
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)
```

이때 사용된 Hyperparameter는 로그에서 아래처럼 확인합니다. 

```python
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              gamma=0, gpu_id=-1, importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_delta_step=0, max_depth=6, min_child_weight=1, missing=nan,
              monotone_constraints='()', n_estimators=100, n_jobs=16,
              num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
```              

아래와 같이 confusion matrix값을 확인합니다. 

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)
cm
```   

이때의 결과는 아래와 같습니다. 

```java
array([[ 60,   5],
       [  7, 116]])
```

score를 구하면 아래와 같습니다. 

```python
print(classifier.score(X_train, y_train))
print(classifier.score(X_test, y_test))
1.0
0.9361702127659575
```

## Reference

[Dataset - sklearn.datasets.load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

[Dataset - Breast_cancer_data.csv](https://github.com/Suji04/ML_from_Scratch/blob/master/Breast_cancer_data.csv)

[Dataset - Breast Cancer Prediction with XGBoost](https://www.kaggle.com/code/armagansarikey/breast-cancer-prediction-with-xgboost/data)

[Breast Cancer Prediction with XGBoost](https://www.kaggle.com/code/armagansarikey/breast-cancer-prediction-with-xgboost)
