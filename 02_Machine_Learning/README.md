# Machine Learning

표 형태 데이터를 중심으로 분류·회귀·불균형 데이터·임계값 정책·공정성 점검·API 입력 검증을 다루는 실험 공간이다. 모델 점수만 비교하지 않고, 데이터 분할과 누수 방지, 운영에서 감당할 수 있는 검토량, 자동화의 한계를 함께 기록한다.

이 문서는 전체 구조와 현재 확인된 결과를 요약한다. 세부 전처리·실행 명령·시각화·원본 수치는 각 하위 README와 results 폴더의 CSV·JSON을 기준으로 확인한다. 원본 데이터와 학습된 모델 파일은 라이선스·용량·개인정보 가능성을 고려해 Git에 포함하지 않는다.

## 한눈에 보기

| 영역 | 문제 | 핵심 방법 | 현재 확인 결과 |
| --- | --- | --- | --- |
| [01 Classification Basics](01_classification-basics/README.md) | Telco 고객 이탈 위험 선별 | Pipeline, Logistic Regression, ROC-AUC, threshold | test Recall **0.807**, ROC-AUC **0.832** |
| [02 Regression Basics](02_regression-basics/README.md) | California Housing 가격 추정 | 회귀 지표, 잔차 분석, Random Forest 비교 | test MAE **31,462**, R² **0.818** |
| [03 Imbalanced Classification](03_imbalanced-classification/README.md) | 희소 신용카드 사기 탐지 | PR-AUC, SMOTE, class weight, threshold | test Recall **0.909**, Precision **0.030** at threshold 0.30 |
| [05 Loan Default](05_featured-loan-default/README.md) | 다음 달 연체 위험 심사 보조 | PR-AUC, 임계값 정책, 공정성 집계, FastAPI | test PR-AUC **0.567**, Recall **0.852** |

기초 노트는 별도 폴더에 둔다. [Basic](Basic/), [LinearRegression](LinearRegression/), [LogisticRegression](LogisticRegression/), [Validation](Validation/)은 특징 학습, 단순·다중 선형 회귀, 로지스틱 회귀, 교차 검증 개념을 확인하는 노트북이며, 위의 재현 가능한 프로젝트 코드와 역할을 구분한다.

## 전체 구조

~~~text
02_Machine_Learning/
├── Basic/                         # 특징 학습 기초 노트북
├── LinearRegression/              # 단순·다중 선형 회귀 노트북
├── LogisticRegression/            # 로지스틱 회귀 노트북
├── Validation/                    # 교차 검증 노트북
├── 01_classification-basics/      # 고객 이탈 이진 분류
├── 02_regression-basics/          # California Housing 회귀
├── 03_imbalanced-classification/  # 신용카드 사기 탐지
├── 05_featured-loan-default/      # 연체 위험 심사 보조 API
└── README.md                       # 전체 목적, 결과, 공통 원칙
~~~

프로젝트 폴더에서 사용하는 공통 경로의 역할은 아래와 같다.

| 경로 | 역할 | Git 관리 원칙 |
| --- | --- | --- |
| src/ | 데이터 준비, 학습, 평가, 공정성 집계 등 실행 코드 | 포함 |
| app/ | FastAPI endpoint, schema, 모델 로드 코드 | 포함 |
| tests/ | 학습·입력 검증·정책 경계값 테스트 | 포함 |
| results/ | 모델 비교, holdout 성능, threshold·공정성 결과 | 포함 |
| assets/ | ROC, 혼동행렬, 잔차, feature importance 시각화 | 포함 |
| data/raw/, data/processed/ | 원본 데이터와 정제 데이터 | 제외 |
| models/ | 직렬화된 모델 가중치 | 제외 |

## 공통 설계 원칙

### 1. train, validation, test의 역할을 나눈다

모델 학습은 train에서만 수행한다. 모델 종류·하이퍼파라미터·운영 임계값은 validation에서 선택하고, test set은 선택을 마친 뒤 최종 성능을 한 번 확인하는 데 사용한다. test 결과를 보고 설정을 반복 조정하면 holdout 성능이 낙관적으로 보일 수 있으므로 피한다.

### 2. 전처리는 학습 데이터에만 fit한다

결측치 처리, 범주형 인코딩, StandardScaler, SMOTE 같은 변환은 Pipeline 또는 학습 파이프라인 안에서 train 데이터로만 학습한다. validation·test 데이터에 전체 데이터 통계나 oversampling 결과가 새어 들어가지 않게 하는 것이 기본 전제다.

### 3. 문제 비용에 맞는 지표와 임계값을 쓴다

정확도가 높아도 실제 가치가 낮을 수 있다. 이탈·사기·연체처럼 양성 비율이 낮거나 놓치는 비용이 큰 문제에서는 Recall, Precision, PR-AUC와 threshold별 검토 대상 수를 함께 본다. 회귀에서는 MAE, RMSE, R²뿐 아니라 구간별 오차와 잔차 패턴을 확인한다.

### 4. 예측은 자동 결정을 뜻하지 않는다

연체 위험 API는 심사 담당자의 우선순위 판단을 돕는 용도이며, 자동 승인·거절 시스템이 아니다. 임계값은 조직이 감당할 수 있는 추가 심사량, 누락 비용, 고객 영향에 따라 달라진다. 현재 수치는 데이터셋 안의 검증 결과이지 실제 정책 효과의 보장은 아니다.

### 5. 집단별 차이는 관찰·점검하되 단정하지 않는다

공정성 리포트는 성별 코드·연령대별 실제 양성률, 예측 양성률, Recall, FPR을 집계한다. 집단 간 차이는 데이터 분포와 표본 수의 영향을 받으므로, 이를 차별 여부나 인과관계의 결론으로 해석하지 않는다. 민감 특성은 감사 지표로 다루고 자동 의사결정 근거로 사용하지 않는다.

~~~text
원본 데이터
  → 식별자·타깃 누수·결측치 점검
  → train / validation / test 분할
  → Pipeline 내부 전처리·학습
  → validation 모델 비교·threshold 선택
  → holdout test 최종 평가
  → 오류·집단별 지표·검토량 해석
  → 필요 시 FastAPI 심사 보조 응답으로 연결
~~~

## 프로젝트별 내용과 결과

### 01. Classification Basics — Telco 고객 이탈 예측

고객의 이탈 확률을 추정해 유지 캠페인 검토 대상을 선별한다. 데이터는 stratify를 적용해 train 60% / validation 20% / test 20%로 분리했고, 전처리와 스케일링은 Pipeline 내부에서 train 데이터로만 fit했다.

| 항목 | 최종 holdout test 결과 |
| --- | ---: |
| 모델 | Logistic Regression |
| Precision | 0.508 |
| Recall | **0.807** |
| F1 | 0.624 |
| ROC-AUC | **0.832** |
| 분할 크기 | train 4,225 / validation 1,409 / test 1,409 |

이탈 고객을 놓치는 비용이 더 크다는 가정에서 Recall을 우선했다. 다만 Recall을 높이는 정책은 더 많은 고객을 캠페인 후보로 포함할 수 있으므로, 단순히 높은 점수로 끝내지 않고 임계값별 대상 수·Precision·Recall을 threshold-policy.csv에 남긴다.

장기 계약은 이탈 위험을 낮추고 월 요금 부담·전자 수표 결제·지원 서비스 부재는 위험을 높이는 경향을 보였다. 이는 데이터의 연관성에 대한 설명일 뿐 할인·지원 정책의 효과를 뜻하지 않는다. 실제 정책 효과는 대조군을 둔 실험으로 검증해야 한다.

~~~bash
cd 01_classification-basics
pip install -r requirements.txt
python src/download_data.py
python src/train.py
python src/evaluate.py
~~~

ROC 곡선·혼동행렬·특성 영향도는 [프로젝트 README](01_classification-basics/README.md)의 assets에서, 최종 수치는 [test-metrics.json](01_classification-basics/results/test-metrics.json)에서 확인한다.

### 02. Regression Basics — California Housing 가격 예측

California Housing 데이터에서 주택 가격을 추정하며, 평균값 기준선·선형 회귀·Gradient Boosting·Random Forest를 같은 test set에서 비교한다. 결과 해석은 실제 가격 단위로 되돌린 MAE와 잔차를 중심으로 한다.

| 모델 | Test MAE | Test RMSE | Test R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Dummy mean | 90,607 | 114,486 | -0.000 | 62.89% |
| Linear Regression | 50,670 | 70,059 | 0.625 | 29.19% |
| Gradient Boosting | 38,278 | 55,903 | 0.762 | 21.50% |
| Random Forest | **31,462** | **48,787** | **0.818** | **17.62%** |

Random Forest는 평균값 기준선보다 MAE를 크게 낮췄지만, 이 수치가 개별 매물 가격의 보장 범위를 뜻하지는 않는다. 고가 주택의 가격 상한과 데이터가 나타내는 지역·시기의 한계 때문에 최고 가격대 오차, 가격 구간별 MAE, ocean_proximity 범주별 오차를 함께 점검해야 한다.

~~~bash
cd 02_regression-basics
pip install -r requirements.txt
python src/download_data.py
python src/train.py
~~~

모델 비교는 [model-comparison.csv](02_regression-basics/results/model-comparison.csv), 잔차·target 분포·모델 비교 그래프는 assets에서 확인한다. 다음 단계는 같은 교차 검증 조건에서 원본 target과 log1p target 변환을 비교하고, 5-fold MAE 평균·표준편차를 추가하는 것이다.

### 03. Imbalanced Classification — 신용카드 사기 탐지

284,807건의 거래 중 사기 거래는 492건(약 0.17%)인 강한 불균형 데이터다. 모든 거래를 정상으로 예측해도 accuracy가 약 99.83%가 되므로, accuracy를 모델 선택 기준으로 쓰지 않는다. 기본 Logistic Regression, class-weight Logistic Regression, SMOTE Logistic Regression을 PR-AUC·Recall·Precision으로 비교하고, SMOTE는 train 데이터에만 적용한다.

| 모델 | validation PR-AUC | validation Recall | validation Precision |
| --- | ---: | ---: | ---: |
| 기본 Logistic Regression | 0.6705 | 0.6327 | **0.8158** |
| class-weight Logistic Regression | 0.6857 | **0.8776** | 0.0650 |
| SMOTE Logistic Regression | **0.6857** | **0.8776** | 0.0622 |

현재 선택은 validation PR-AUC가 가장 높은 SMOTE Logistic Regression이며, validation에서 높은 Recall을 우선해 threshold 0.30을 선택했다. 최종 holdout test에서 PR-AUC 0.7636, Recall 0.9091, Precision 0.0302를 기록했다.

| threshold | validation 심사 대상 | Recall | Precision | 해석 |
| ---: | ---: | ---: | ---: | --- |
| 0.30 | 2,946건 | 0.9082 | 0.0302 | 사기 누락을 줄이지만 오탐 검토량이 큼 |
| 0.50 | 1,383건 | 0.8776 | 0.0622 | 검토량과 탐지율의 중간안 |
| 0.70 | 701건 | 0.8571 | 0.1198 | 더 적은 건을 보되 더 많은 사기를 놓칠 수 있음 |

낮은 threshold에서의 매우 낮은 precision은 “탐지 성공”만으로 배포할 수 없다는 근거다. 실제 운영에서는 건당 수동 심사 비용, 사기 미탐 비용, 조사 가능 인력을 수치화해 threshold를 다시 선택해야 한다. 현재 데이터는 학습 목적의 공개 데이터이며 실제 거래 차단·승인 정책으로 바로 연결하지 않는다.

~~~bash
cd 03_imbalanced-classification
pip install -r requirements.txt
python src/train.py
~~~

세부 결과는 [model-comparison.csv](03_imbalanced-classification/results/model-comparison.csv), [threshold-policy.csv](03_imbalanced-classification/results/threshold-policy.csv), [final-test.json](03_imbalanced-classification/results/final-test.json)에 남아 있다.

### 05. Featured Loan Default — 연체 위험 심사 보조 API

UCI Default of Credit Card Clients 30,000건을 사용해 다음 달 연체 위험을 추정한다. 목표는 위험 신청자를 우선 검토해 심사 자원을 배분하는 것이며, 자동 승인·거절을 수행하지 않는다. 식별자 ID와 예측 시점 이후 정보·타깃은 입력에서 제외하고, StandardScaler와 SMOTE는 학습 파이프라인 안에서만 적용한다.

validation PR-AUC를 1차 모델 선택 기준으로 비교했다.

| 모델 | validation PR-AUC | Recall | Precision | 역할 |
| --- | ---: | ---: | ---: | --- |
| Random Forest | **0.530** | 0.569 | 0.496 | 최종 후보 |
| 기본 Logistic Regression | 0.478 | 0.226 | **0.661** | 비교 기준선 |
| SMOTE Logistic Regression | 0.478 | **0.615** | 0.363 | 비교 |
| class-weight Logistic Regression | 0.476 | 0.607 | 0.373 | 비교 |

최종 정책은 연체 누락 비용을 우선한다는 가정 아래 threshold 0.30을 선택했다.

| 항목 | 결과 |
| --- | ---: |
| 최종 모델 | Random Forest |
| 최종 test PR-AUC | **0.5668** |
| 최종 test Recall | **0.8515** |
| 최종 test Precision | 0.3200 |
| validation 심사 대상 수 (threshold 0.30) | 3,586명 |
| validation Recall / Precision (threshold 0.30) | 0.8538 / 0.3160 |

threshold 0.30은 미탐을 줄이는 대신 더 많은 추가 심사를 만든다. threshold 0.50에서는 1,521명을 검토하고 Recall 0.5690·Precision 0.4964, threshold 0.70에서는 781명을 검토하고 Recall 0.3640·Precision 0.6184를 기록했다. 어느 수준이 적절한지는 실제 심사 비용·회수율·고객 영향에 대한 정보가 있어야 결정할 수 있다.

최종 test에서 성별 코드별 예측 양성률은 코드 1이 0.624, 코드 2가 0.565였고 FPR은 각각 0.546, 0.494였다. 60세 이상은 82명으로 표본이 작고 Recall 0.762를 기록했다. 이는 배포 전 점검용 관찰값이며, 차별 또는 인과관계의 판정이 아니다. 성별·연령은 모델 응답의 심사 근거로 노출하지 않고 감사용 집계 지표로만 다룬다.

FastAPI는 POST /predict에서 연체 확률, risk grade, 심사 보조 의견, 적용 threshold, 정책 근거, 참고 위험 신호와 자동 결정 금지 안내를 반환한다. 잘못된 필수 입력과 범위 오류는 422로 거절한다.

~~~bash
cd 05_featured-loan-default
pip install -r requirements.txt
python src/prepare_data.py
python src/train.py
python src/analyze_fairness.py
uvicorn app.main:app --reload
~~~

테스트와 API 계약, 집단별 결과, 데이터 범위·한계는 [프로젝트 README](05_featured-loan-default/README.md), [fairness-report.md](05_featured-loan-default/results/fairness-report.md), [data dictionary](05_featured-loan-default/data/data-dictionary.md)를 따른다.

## 재현성과 결과물 확인

| 확인하려는 내용 | 우선 파일 |
| --- | --- |
| 최종 holdout 성능 | results/test-metrics.json, results/final-test.json, results/metrics.json |
| validation 기반 모델 선택 | results/validation-model-comparison.csv, results/model-comparison.csv |
| threshold별 검토량·Recall·Precision | results/threshold-policy.csv |
| 집단별 관찰 지표 | results/fairness-by-group.csv, results/fairness-report.md |
| 설명·오류 시각화 | assets의 ROC, confusion matrix, residual plot, feature importance |
| API 입력·정책 계약 | tests와 하위 프로젝트 README |

각 프로젝트의 단위 테스트는 해당 폴더에서 실행한다.

~~~bash
python -m unittest discover -s tests -v
~~~

연체 위험 프로젝트는 정상 예측 응답, 필수 입력 누락, 입력 범위 위반, threshold 경계값의 risk grade·정책 설명, 공정성 집계 계산을 테스트한다. 테스트가 통과해도 데이터 대표성·정책의 적절성·외부 환경 성능까지 보장하지는 않는다.

## 데이터·모델 파일 관리

- 원본 데이터는 각 프로젝트의 data/raw에 두고 Git에 올리지 않는다.
- 정제 데이터와 모델 직렬화 파일도 용량·라이선스·재현 규칙을 고려해 보통 제외한다.
- 코드, requirements, 데이터 사전, 실행 설정, 작은 결과 CSV·JSON, 시각화, 테스트는 Git에 포함한다.
- 고객·거래·신용 정보처럼 민감할 수 있는 원문·개별 예측·식별자는 예제와 결과물에 노출하지 않는다.
- 원본 데이터를 다시 내려받거나 정제해야 할 때는 하위 README의 데이터 출처·명령을 사용하고, 생성된 결과는 새 실행 환경 정보와 함께 검토한다.

## 다음 정비 우선순위

1. 모든 분류 프로젝트에 seed·라이브러리 버전·실행 환경·분할 요약을 일관된 JSON 형식으로 남긴다.
2. 고객 이탈·사기 탐지·연체 위험 모델에서 임계값을 실제 검토 처리량과 비용 가정으로 연결하고, 선택 근거를 명시한다.
3. California Housing 회귀에는 5-fold 교차 검증, target log 변환 비교, 가격대·지역 범주별 MAE를 추가한다.
4. 불균형 사기 탐지는 precision이 낮은 threshold 정책의 운영 가능성을 별도 검토하고, 시간 순서 또는 더 현실적인 holdout 검증을 추가한다.
5. 연체 위험 API에는 확률 보정, 시간 기준 검증, 데이터 드리프트·집단별 성능 모니터링을 추가한다. 모든 예측은 심사 담당자의 검토를 보조하는 범위에서 사용한다.
