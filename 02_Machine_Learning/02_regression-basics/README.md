# California Housing 가격 예측

공개 중고차 데이터의 재현성과 라이선스 문제를 고려해, 변수 품질이 검증된 California Housing 데이터로 회귀 기본기를 구현했다.

## 현재 결과

Random Forest는 Test MAE 31,462, RMSE 48,787, R2 0.818로 평균값 기준 모델의 MAE 90,607보다 크게 개선됐다.

## 다음 검증 기준

- 원본 가격과 log1p 가격 변환을 같은 교차 검증 조건에서 비교한다.
- 5-Fold CV의 MAE 평균과 표준편차를 기록한다.
- 저가·중가·고가 구간 및 ocean_proximity 범주별 MAE를 비교한다.
- 고가 주택 가격 상한이 존재하므로, 최고 가격대의 오차와 편향 가능성을 별도 해석한다.

## 실행

~~~powershell
pip install -r requirements.txt
python src/download_data.py
python src/train.py
~~~

예측 결과는 실제 가격 단위로 되돌린 뒤 MAE와 함께 해석한다. 예측값은 의사결정 참고치이며, Test MAE 범위는 보장 구간이 아니다.
