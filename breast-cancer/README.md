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

## xgboost로 분석하기 

## Reference

[Dataset - sklearn.datasets.load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

[Dataset - Breast_cancer_data.csv](https://github.com/Suji04/ML_from_Scratch/blob/master/Breast_cancer_data.csv)

[Dataset - Breast Cancer Prediction with XGBoost](https://www.kaggle.com/code/armagansarikey/breast-cancer-prediction-with-xgboost/data)

[Breast Cancer Prediction with XGBoost](https://www.kaggle.com/code/armagansarikey/breast-cancer-prediction-with-xgboost)
