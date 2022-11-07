# XGBoost Algorithm

[XGBoost](https://github.com/kyopark2014/ML-Algorithms/blob/main/xgboost.md)는 타깃(Target)과 모델의 예측 사이에 손실 함수를 정의하여, 여러 개의 약한 예측 모델을 순차적으로 구축하여 반복적으로 오차를 개선하면서, 하나의 강한 예측 모델을 만드는 [Boosting](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md)방식으로 [앙상블 기법](https://github.com/kyopark2014/ML-Algorithms/blob/main/ensemble.md)의 하나입니다. [기본 학습기](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#basic-learner)로 주로 옅은 Depth의 [결정트리](https://github.com/kyopark2014/ML-Algorithms/blob/main/decision-tree.md)를 활용하여, [손실 함수](https://github.com/kyopark2014/ML-Algorithms/blob/main/loss-function.md)을 계산하여 [경사 하강법(Gradient desecent)](https://github.com/kyopark2014/ML-Algorithms/blob/main/stochastic-gradient-descent.md#gradient-descent)을 사용하여 [잔차(Residual)](https://github.com/kyopark2014/ML-Algorithms/blob/main/boosting.md#residual)을 최소화하는 방향으로 최적화를 수행합니다. 


## Breast Cancer 분석하기

[Breast cancer 분석](https://github.com/kyopark2014/ML-xgboost/blob/main/breast-cancer)에서는 XGBoost로 Breast cancer 분석하는 과정을 설명합니다. 


## 보험사기 분석하기


[자동차 보험 사기 검출](https://github.com/kyopark2014/ML-xgboost/tree/main/auto-insurance-claim)에서는 XGBoost로 자동차 보험 사기 검출을 분석하는 과정을 설명합니다. 

## Wine Quality 측정 

[Wine Quality](https://github.com/kyopark2014/ML-xgboost/tree/main/wine-quality)에서는 XGBoost를 이용한 회귀(Regression) 문제를 설명하고 있습니다.  



## Reference

[Amazon SageMaker 모델 학습 방법 소개 - AWS AIML 스페셜 웨비나](https://www.youtube.com/watch?v=oQ7glJfD-BQ&list=PLORxAVAC5fUULZBkbSE--PSY6bywP7gyr)

[SageMaker 스페셜 웨비나 - Github](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/sagemaker/sm-special-webinar)

[Dataset - Architect and build the full machine learning lifecycle with AWS: An end-to-end Amazon SageMaker demo](https://aws.amazon.com/ko/blogs/machine-learning/architect-and-build-the-full-machine-learning-lifecycle-with-amazon-sagemaker/)

[XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)


[XGBoost와 사이킷런을 활용한 그레이디언트 부스팅 - 한빛 미디어](https://github.com/rickiepark/handson-gb)
