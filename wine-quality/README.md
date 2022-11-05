# Wine Quality (Regression)

[XGBoost를 이용한 Wine Quality](https://github.com/kyopark2014/ML-Algorithms/tree/main/kaggle/xgboost-wine-quality)에서는 [XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)를 이용하여 [Regression](https://github.com/kyopark2014/ML-Algorithms/blob/main/regression.md)을 수행하였습니다. [xgboost-wine-quality.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/xgboost-wine-quality.ipynb)은 Jupyter Notebook 파일로서, 데이터 전처리 및 XGBoost algotirhm에 대한 Hyperparameter Optimization을 수행합니다. 본격적인 학습을 수행하기 위해서는 jupyter notebook으로 검증된 알고리즘을 python으로 변환하여야 합니다. 

## Tranining

[ML 알고리즘을 Python 코드로 변환](https://github.com/kyopark2014/ML-Algorithms/blob/main/python-translation.md)에 따라서 jupyter notebook 파일을 [xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/xgboost-wine-quality.py)로 변환할 수 있습니다. 아래는 상세 동작을 설명합니다. 

1) 확장자가 ipyb인 jupyter notebook 파일을 아래 명령어를 이용하여 python 파일로 변환 합니다. 

아래 명령어로 [xgboost-wine-quality.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/xgboost-wine-quality.ipynb)에서 [step0-xgboost-wine-quality.py ](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step0-xgboost-wine-quality.py)을 생성합니다.

```java
jupyter nbconvert jupyter nbconvert xgboost-wine-quality.ipynb --to script --output step0-xgboost-wine-quality
```


2) 불필요한 코드 정리

jupyter notebook에서 데이터의 구조를 이해하기 위해 사용했던 코드들은 본격적인 학습에서는 사용되지 않습니다. 따라서, [step1-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step1-xgboost-wine-quality.py)와 같이 불필요한 코드를 삭제합니다. 

3) Python 함수로 리팩터링

함수 형태로 refactoring을 하면, 코드를 읽기 쉬워지고 유지 관리가 용이해집니다. [step2-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step2-xgboost-wine-quality.py)와 같이 함수로 변환합니다.
이때, main은 진입점(entry point)이므로 실행중인지 여부를 확인하여 아래처럼 사용합니다. 

```python
if __name__ == '__main__':
    main()
```


4) jupyter notebook이 python 코드로 변환되었으므로 아래와 같이 학습을 수행합니다. [xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/xgboost-wine-quality.py)는 학습한 결과를 [xgboost_wine_quality.json](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/xgboost_wine_quality.json)로 저장하고, 추론 시험을 위해 [samples.json](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/samples.json)을 생성합니다. 

```python
python3 xgboost-wine-quality.py
```


## Inference

[xgboost-wine-quality-inference.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/xgboost-wine-quality-inference.ipynb)에서 predict()을 이용하여 inference 동작을 시험하였습니다. [inference.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/inference.py)와 같이 event에서 body를 추출하여 추론(inference)을 할 수 있도록 하였습니다. 

추론 동작이 잘 동작하는것을 확인하기 위하여 [inference-test.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/inference-test.py)에서는 [samples.json](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/samples.json)을 로드하여 [inference.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/inference.py)의 handler를 호출합니다. 추론 동작은 아래와 같이 확인 할 수 있습니다. 

```python
python3 inference-test.py
```
