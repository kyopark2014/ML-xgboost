# Wine Quality (Regression)

[XGBoost를 이용한 Wine Quality](https://github.com/kyopark2014/ML-Algorithms/tree/main/kaggle/xgboost-wine-quality)에서 구현한 [xgboost-wine-quality.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/xgboost-wine-quality.ipynb)는 Jupyter Notebook 파일로서, 데이터의 전처리 및 XGBoost algotirhm에 대한 Hyperparameter Optimization을 수행할 수 있습니다. 하지만 이를 실제 사용하기 위해서는 python 파일로 변경하여야 합니다. 

## Python 코드로 변환 

[ML 알고리즘을 Python 코드로 변환](https://github.com/kyopark2014/ML-Algorithms/blob/main/python-translation.md)에 따라 python 코드로 변환합니다.


1) 확장자가 ipyb인 jupyter notebook 파일을 아래 명령어를 이용하여 python 파일로 변환 합니다. 


https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step0-xgboost-wine-quality.py 

jupyter nbconvert jupyter nbconvert xgboost-wine-quality.ipynb --to script --output step0-xgboost-wine-quality

[xgboost-wine-quality.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/xgboost-wine-quality.ipynb)을 상기 명령어로 변환하면, [step0-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step0-xgboost-wine-quality.py)와 같이 python 파일로 변환할 수 있습니다. 

2) 불필요한 코드 정리

jupyter notebook에서 데이터의 구조를 이해하고, 도표를 작성할때 사용했던 코드들은 본격적인 학습에서는 사용되지 않습니다. 따라서, [step1-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step1-xgboost-wine-quality.py)와 같이 불필요한 코드를 삭제합니다. 

3) Python 함수로 리팩터링

함수로 변경하면 refactoring이 쉬워지고 유지 관리가 수월하여 지므로 [step2-xgboost-wine-quality.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/step2-xgboost-wine-quality.py)와 같이 함수로 변환합니다.
이때, main은 진입점(entry point)이므로 실행중인지 여부를 확인하여 아래처럼 사용합니다. 

```python
if __name__ == '__main__':
    main()
```

## Inference

[xgboost-wine-quality-inference.ipynb](https://github.com/kyopark2014/ML-Algorithms/blob/main/kaggle/xgboost-wine-quality/xgboost-wine-quality-inference.ipynb)을 참조하여 inference를 위한 python을 [xgboost-wine-quality-inference.py](https://github.com/kyopark2014/ML-xgboost/blob/main/wine-quality/src/xgboost-wine-quality-inference.py)과 같이 생성합니다 

