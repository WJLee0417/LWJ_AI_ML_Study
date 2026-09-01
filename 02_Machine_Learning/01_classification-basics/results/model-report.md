# 고객 이탈 모델 결과

## 홀드아웃 테스트

- model: Logistic Regression
- precision: 0.508
- recall: 0.807
- f1: 0.624
- roc_auc: 0.832

이탈 고객을 놓치는 비용이 더 크다는 가정에서 validation recall을 우선해 모델을 선택했다. 전처리·스케일링은 Pipeline 내부에서 train 데이터로만 fit했다.
