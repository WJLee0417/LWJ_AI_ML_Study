# LWJ AI & Machine Learning Study

데이터를 정제하고 분석 근거를 만든 뒤, 머신러닝·딥러닝 모델을 검증하고 API 계약까지 연결하는 과정을 기록한 저장소입니다. 각 프로젝트는 결과 점수만 제시하지 않습니다. 데이터가 어떤 범위에서 수집됐는지, train·validation·test를 어떻게 나눴는지, 모델 선택과 운영 임계값을 어떤 근거로 정했는지, 결과를 어디까지 해석할 수 있는지를 함께 남깁니다.

> **핵심 관점:** 좋은 모델은 높은 점수만으로 완성되지 않습니다. 데이터 품질, 누수 방지, 재현 가능한 실험, 검토 가능한 결과물, 그리고 자동화가 적절하지 않은 경우 사람에게 넘기는 정책까지 갖춰야 합니다.

## 3분 안내

| 보고 싶은 내용 | 바로가기 | 확인할 수 있는 것 |
| --- | --- | --- |
| 데이터가 분석 가능한 상태인지 | [Data Analysis](01_Data_Analysis/README.md) | 중복·결측·취소 주문 처리, EDA, SQL 대조, RFM, 상권 비교 |
| 표 형태 예측 문제를 어떻게 다뤘는지 | [Machine Learning](02_Machine_Learning/README.md) | 분류·회귀·불균형 데이터·임계값 정책·공정성 점검·심사 보조 API |
| 이미지·텍스트 모델을 어떻게 검증했는지 | [Deep Learning](03_Deep_Learning/README.md) | CNN 비교, 전이학습, 한국어 Transformer, 확률 보정, FastAPI 서빙 계약 |
| 서비스 연결을 어떻게 설계했는지 | [Model Serving](03_Deep_Learning/05_model-serving/README.md) | 모델 manifest, 입력 검증, 단일·배치 추론, Spring BFF 연동 기준 |

## 저장소가 다루는 흐름

~~~text
원본 데이터
  → 데이터 품질·관측 단위·제외 규칙 확인
  → 탐색적 분석과 SQL 집계·대조
  → 기준선 모델과 후보 모델 비교
  → validation 기반 모델·임계값·보정 선택
  → holdout test 최종 평가
  → 결과 CSV·JSON·시각화·manifest 기록
  → FastAPI 예측과 사람 검토 정책 연결
~~~

각 단계는 독립된 실습이 아니라 다음 단계의 전제가 된다. 예를 들어 Online Retail 정제 규칙은 Olist SQL 분석의 고객·주문 단위 구분과 이어지고, RFM 세분화는 이탈 위험을 활용한 캠페인 설계의 틀을 제공한다. 이미지·텍스트 모델은 validation에서 확률을 보정하고 review threshold를 고정한 뒤, API가 같은 모델 버전·클래스 순서·전처리를 사용하도록 manifest로 연결한다.

## 전체 구조

~~~text
LWJ_AI_ML_Study/
├── 01_Data_Analysis/       # 정제, EDA, SQL, 세분화, 상권 대시보드
├── 02_Machine_Learning/    # 분류, 회귀, 불균형 학습, 심사 보조 API
├── 03_Deep_Learning/       # PyTorch, CNN, 전이학습, NLP, 모델 서빙
├── .github/workflows/      # MNIST 단위 테스트·smoke test CI
└── README.md
~~~

| 영역 | 중심 질문 | 대표 산출물 |
| --- | --- | --- |
| Data Analysis | 데이터가 믿을 만한가? 무엇을 추가로 확인해야 하는가? | 정제 테이블, 품질 보고서, SQL 결과, 분석 보고서, 대시보드 |
| Machine Learning | 어떤 모델과 임계값이 운영 목표에 맞는가? | Pipeline, holdout 성능, threshold 정책, 공정성 집계, FastAPI 응답 |
| Deep Learning | 공간·문맥 정보를 쓰는 모델의 이득과 비용은 무엇인가? | 학습 곡선, 혼동행렬, Grad-CAM, calibration, model manifest |

## 대표 결과

아래 수치는 각 프로젝트에서 고정된 데이터 분할과 실행 조건으로 생성한 결과다. 서로 다른 데이터셋·문제·지표를 단순 순위로 비교하지 않으며, 각 수치의 세부 조건과 한계는 링크된 프로젝트 문서에서 확인한다.

### 데이터 분석

| 프로젝트 | 데이터·질문 | 확인 결과 | 해석 범위 |
| --- | --- | --- | --- |
| [Online Retail 정제](01_Data_Analysis/01_data-cleaning/README.md) | 541,909개 주문 상품 라인에서 품질 문제는 무엇인가 | 완전 중복 5,268행, 고객 ID 결측 135,080행, 취소 주문 10,587행을 식별하고 목적별 테이블 생성 | 취소·결측을 무조건 삭제하지 않고 분석 목적별 포함 범위를 분리 |
| [서울 공공자전거 EDA](01_Data_Analysis/02_eda-seoul-bike/README.md) | 시간·요일·계절·강수량별 수요는 어떤가 | 18시 전체 피크, 평일 18시·주말 17시 피크, 5mm 이상 강수 시 평균 수요 약 90.4% 감소 | 단일 연도·시간 단위 집계이므로 대여소별 재배치 위치나 인과 효과는 확정하지 않음 |
| [Olist SQL 분석](01_Data_Analysis/03_sql-analysis/README.md) | 매출·고객·상품 지표를 관계형 모델에서 검증할 수 있는가 | 10개 SQL, 344,483행 적재, 23개월 월별 매출을 Pandas와 대조해 오차 0.00 | 주문 중심 원본이므로 CRM 회원 전체의 행동으로 일반화하지 않음 |
| [고객 세분화](01_Data_Analysis/04_customer-segmentation/README.md) | RFM 고객군을 실행 가능한 수준으로 나눌 수 있는가 | 93,358명에서 K=2 선택, 80% 표본·10개 seed 평균 ARI 0.991 | 군집은 행동 요약이며, 혜택 효과는 A/B 테스트로 검증 필요 |
| [강남구 카페업 대시보드](01_Data_Analysis/05_commercial-district-dashboard/README.md) | 매출·점포·유동인구로 추가 검토 상권을 좁힐 수 있는가 | 2025 Q4에 유동인구는 높고 점포당 추정매출은 낮은 후보 13곳 식별 | 창업 추천이 아니라 임대료·경쟁·시간대 수요를 더 확인할 후보군 |

### 머신러닝

| 프로젝트 | 모델 선택·정책 | 최종 결과 | 해석과 제한 |
| --- | --- | --- | --- |
| [Telco 고객 이탈](02_Machine_Learning/01_classification-basics/README.md) | Logistic Regression, 이탈 누락 비용을 고려해 Recall 우선 | test Recall **0.807**, Precision 0.508, F1 0.624, ROC-AUC 0.832 | 캠페인 효과나 특성의 인과 효과는 대조군 실험 필요 |
| [California Housing 회귀](02_Machine_Learning/02_regression-basics/README.md) | 평균 기준선·선형 회귀·Boosting·Random Forest 비교 | Random Forest test MAE **31,462**, RMSE 48,787, R² **0.818** | 개별 주택 가격 보장 범위가 아니며, 고가 주택 상한·구간별 오차 점검 필요 |
| [신용카드 사기 탐지](02_Machine_Learning/03_imbalanced-classification/README.md) | SMOTE Logistic Regression, PR-AUC·Recall·Precision·검토량 비교 | threshold 0.30 test PR-AUC 0.7636, Recall **0.9091**, Precision 0.0302 | 높은 Recall의 대가로 오탐 검토량이 커 실제 심사 비용에 따른 threshold 재선택 필요 |
| [연체 위험 심사 보조](02_Machine_Learning/05_featured-loan-default/README.md) | Random Forest, validation PR-AUC 기준 선택, threshold 0.30 | test PR-AUC **0.5668**, Recall **0.8515**, Precision 0.3200 | 자동 승인·거절이 아닌 심사 우선순위 보조이며, 집단별 차이를 감사 지표로 점검 |

### 딥러닝

| 프로젝트 | 모델·실험 | 최종 결과 | 해석과 제한 |
| --- | --- | --- | --- |
| [MNIST FCN vs CNN](03_Deep_Learning/02_mnist-cnn-benchmark/README.md) | 같은 분할에서 FCN·CNN 비교 | CNN test accuracy·macro F1 **0.9869**, FCN은 0.9713·0.9711 | CPU batch 1에서는 CNN이 더 느렸으므로 정확도와 추론 비용을 분리해 해석 |
| [RealWaste 전이학습](03_Deep_Learning/03_transfer-learning/README.md) | ResNet18 fine-tuning + augmentation + class weight | test macro F1 **0.9297**, accuracy 0.9233, Temperature Scaling 후 ECE 0.0188 | 외부 조명·배경·촬영 기기에서는 별도 평가 필요, Grad-CAM은 보조 해석 도구 |
| [한국어 문의 분류](03_Deep_Learning/04_text-classification/README.md) | TF-IDF 기준선과 KLUE-BERT fine-tuning 비교 | holdout 9,000건 macro F1 **0.9551**, 자동 처리 precision 95.52% | 모델 크기 443.5 MB·학습 약 42분 49초, 새 채널·시기 데이터로 재검증 필요 |
| [이미지 모델 서빙](03_Deep_Learning/05_model-serving/README.md) | FastAPI, manifest 검증, 단일·배치 추론 | health, model-info, predict, predict/batch와 입력 검증 구현 | 실제 checkpoint를 연결한 end-to-end 지연·부하 수치는 별도 측정 필요 |

## 문제를 다루는 방식

### 데이터 품질과 누수 방지

- 중복, 결측, 취소·반품, 식별자, 예측 시점 이후 정보처럼 결과를 왜곡할 수 있는 요소를 먼저 점검합니다.
- train·validation·test는 역할을 분리합니다. 모델·epoch·임계값은 validation으로 선택하고, test는 최종 평가에만 사용합니다.
- 이미지의 연속 촬영본은 group_id, 문의 데이터는 상담번호를 기준으로 분할해 유사 샘플이 train과 test에 섞이는 누수를 줄입니다.
- 전처리·스케일러·SMOTE는 train 데이터에서만 fit합니다. 전체 데이터 통계가 validation·test로 새어 들어가지 않게 Pipeline을 사용합니다.

### 지표와 운영 정책

- 불균형 문제는 accuracy 하나로 판단하지 않고 PR-AUC, Recall, Precision, F1, threshold별 검토 대상 수를 함께 기록합니다.
- 회귀는 MAE, RMSE, R²와 잔차·가격 구간·범주별 오차를 함께 확인합니다.
- 딥러닝 분류는 클래스별 F1, 혼동행렬, 오분류 샘플, 파라미터 수, checkpoint 크기, 학습·추론 시간을 함께 봅니다.
- confidence는 사실상의 정답 보장이 아닙니다. validation logits로 Temperature Scaling을 적용하고 NLL·ECE를 확인한 뒤 review threshold를 고정합니다.
- review threshold를 넘지 못한 예측은 needs_review로 반환해 자동 확정이 아니라 사람 검토 흐름으로 보냅니다.

### 재현 가능한 산출물

| 산출물 | 역할 |
| --- | --- |
| README와 data dictionary | 문제 정의, 데이터 범위, 제외 규칙, 한계 |
| src와 config | 동일한 전처리·학습·평가의 재실행 |
| results의 CSV·JSON·Markdown | 수치 근거, 모델·임계값 비교, 검증 보고서 |
| assets | ROC, 혼동행렬, 잔차, 학습 곡선, Grad-CAM, 대시보드 화면 |
| tests | 입력 shape, 데이터 분할, API validation, 정책 경계값 검증 |
| model manifest | 모델 버전, 클래스 순서, 전처리, calibration, threshold 계약 |

## 기술 구성

| 범주 | 사용 기술 | 적용 사례 |
| --- | --- | --- |
| 분석·정제 | Python, Pandas, NumPy, Jupyter | 주문 정제, 공공자전거 EDA, 상권 지표 |
| 관계형 분석 | MySQL 8, Docker Compose, SQL CTE·Window Function | Olist 주문·고객·상품 적재와 월별 매출 대조 |
| 머신러닝 | scikit-learn, imbalanced-learn, Pipeline, SMOTE | 이탈·회귀·사기·연체 위험 모델 |
| 딥러닝 | PyTorch, torchvision, Transformers | MNIST CNN, ResNet18, KLUE-BERT |
| 시각화·앱 | Matplotlib, Seaborn, Plotly, Streamlit | EDA·모델 해석·상권 대시보드 |
| API·연동 | FastAPI, Pydantic, Docker, Spring BFF 계약 | 이미지 추론 endpoint와 입력·응답 검증 |
| 품질 관리 | unittest, GitHub Actions | MNIST의 단위 테스트·데이터 다운로드 없는 학습 smoke test |

## 빠른 탐색 순서

### 분석에서 의사결정 근거까지

1. [Online Retail 정제](01_Data_Analysis/01_data-cleaning/README.md)에서 분석 대상 테이블을 만드는 품질 규칙을 확인합니다.
2. [서울 자전거 EDA](01_Data_Analysis/02_eda-seoul-bike/README.md)에서 집계·시각화를 운영 가설로 번역하는 방식을 봅니다.
3. [Olist SQL 분석](01_Data_Analysis/03_sql-analysis/README.md)에서 관계형 적재, SQL, Pandas 대조를 확인합니다.
4. [고객 세분화](01_Data_Analysis/04_customer-segmentation/README.md)에서 K 선택·안정성·캠페인 설계의 한계를 확인합니다.
5. [상권 대시보드](01_Data_Analysis/05_commercial-district-dashboard/README.md)에서 여러 지표를 결합해 추가 검토 후보를 좁히는 인터페이스를 확인합니다.

### 예측 모델에서 서빙 계약까지

1. [고객 이탈](02_Machine_Learning/01_classification-basics/README.md)과 [사기 탐지](02_Machine_Learning/03_imbalanced-classification/README.md)에서 threshold가 검토량과 어떻게 교환되는지 확인합니다.
2. [연체 위험 심사 보조](02_Machine_Learning/05_featured-loan-default/README.md)에서 PR-AUC·집단별 지표·FastAPI 입력 검증을 확인합니다.
3. [MNIST 벤치마크](03_Deep_Learning/02_mnist-cnn-benchmark/README.md)에서 설명형 노트북을 재현 가능한 학습·평가 코드로 분리한 구조를 봅니다.
4. [전이학습](03_Deep_Learning/03_transfer-learning/README.md)과 [문의 분류](03_Deep_Learning/04_text-classification/README.md)에서 모델 비교, calibration, 자동 처리·검토 정책을 확인합니다.
5. [모델 서빙](03_Deep_Learning/05_model-serving/README.md)에서 학습 모델·전처리·threshold가 API와 일치하도록 manifest를 검증하는 방식을 봅니다.

## 실행과 검증

프로젝트마다 의존성과 데이터 출처가 다르므로, 먼저 대상 폴더의 README를 확인한 뒤 그 폴더에서 실행합니다. 일반적인 순서는 아래와 같습니다.

~~~powershell
cd 03_Deep_Learning\02_mnist-cnn-benchmark
pip install -r requirements.txt
python src/train.py --model both --epochs 5
python src/evaluate.py --model both
python -m unittest discover -s tests -v
~~~

MNIST 벤치마크는 GitHub Actions에서 단위 테스트와 MNIST 다운로드 없는 한 번의 optimizer step smoke test를 실행합니다. 이미지·텍스트 모델의 전체 학습은 데이터·GPU·모델 다운로드 비용이 커 CI에서 무조건 재실행하지 않습니다. 대신 결과 CSV·JSON·manifest와 단위 테스트를 함께 검토합니다.

GPU가 필요한 딥러닝 학습은 환경에 따라 수치가 미세하게 달라질 수 있습니다. 실행 시 seed, 라이브러리 버전, device, CUDA 사용 여부, batch size, epoch, 학습 시간을 결과 파일에 남기며, 결과가 다를 때는 먼저 이 정보를 비교합니다.

## 데이터·모델·비밀값 관리

- 원본 데이터는 각 프로젝트의 data/raw에 두고 Git에 올리지 않습니다.
- 대용량 정제 데이터, checkpoint, tokenizer는 보통 data/processed 또는 models에 두고 제외합니다.
- 데이터 사전, 다운로드·정제 코드, 작은 결과 파일, 시각화, 실행 설정, 테스트는 재현 근거로 Git에 포함합니다.
- API key, MySQL 비밀번호, 개인 인증키는 환경 변수 또는 개인 설정으로 전달하며 저장소에 저장하지 않습니다.
- 고객 문의·거래·신용 정보처럼 민감할 수 있는 원문과 개별 예측 결과는 예제·오분류 파일에 포함하지 않거나 Git에서 제외합니다.

## 현재 한계와 다음 우선순위

1. 모든 프로젝트에 실행 환경·데이터 버전·분할 요약을 같은 형식의 manifest로 남겨 결과 비교와 재실행을 더 단순하게 만듭니다.
2. 머신러닝 모델의 임계값을 실제 검토 인력·비용·고객 영향과 연결하고, 시간 기준 holdout과 확률 보정을 늘립니다.
3. 이미지·텍스트 모델은 외부 환경·새 채널·미래 시점 데이터에서 성능과 자동 처리 precision을 다시 평가합니다.
4. FastAPI에는 실제 checkpoint를 연결한 end-to-end 지연 시간, 메모리, 배치 처리량, 오류율 측정을 추가합니다.
5. 세분화·상권 후보·특성 중요도에서 나온 가설은 실제 운영 데이터와 대조군을 둔 실험으로 검증하며, 상관관계를 효과로 단정하지 않습니다.

각 하위 README는 이 문서의 요약을 뒷받침하는 실행 코드·결과 파일·한계 설명으로 연결됩니다.
