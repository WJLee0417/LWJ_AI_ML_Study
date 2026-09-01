# 불균형 신용카드 사기 탐지

284,807건 중 사기 거래는 492건(약 0.17%)이다. 모든 거래를 정상으로 예측해도 Accuracy는 약 99.83%이므로 Accuracy를 모델 선택 기준으로 사용하지 않는다.

SMOTE는 Train 데이터에만 적용하고, Validation/Test 데이터에는 원래 불균형 비율을 유지한다. 기본 Logistic Regression, class_weight balanced, SMOTE Logistic Regression을 PR-AUC·Recall·Precision으로 비교한다.

~~~powershell
pip install -r requirements.txt
python src/train.py
~~~

임계값 0.3·0.5·0.7별 탐지 거래 수·Recall·Precision은 results/threshold-policy.csv에 기록한다. 최종 운영 임계값은 최고 점수 대신 추가 심사 처리 가능 건수와 미탐 비용을 고려해 선택한다.
