# California Housing 가격 예측

주택의 위치, 주택 연식, 방 수, 인구, 중위소득, 해안 접근성으로 중위 주택 가격을 예측한다.

## 검증

- Train/Test 80:20 분리
- 평균값 기준, Linear Regression, Random Forest, Gradient Boosting 비교
- MAE, RMSE, R², MAPE를 함께 비교
- 잔차 그래프로 예측 가격대별 반복 오차를 점검

## 실행

~~~powershell
pip install -r requirements.txt
python src/train.py
~~~

산출물은 assets의 타깃 분포·잔차·모델 비교 그래프와 results/model-comparison.csv에 생성된다.
