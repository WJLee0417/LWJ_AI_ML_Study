# 한국어 고객 문의 자동 분류

고객 문의 텍스트를 배송·환불·상품 업무 라벨로 분류하고, 충분히 확신하는 건만 자동 배정하며 나머지는 상담사 검토로 넘기는 모델을 만든다.

> **결론:** 동일한 공식 holdout test 9,000건에서 KLUE-BERT fine-tuning은 TF-IDF 기준선보다 macro F1을 **0.8761 → 0.9551**로 높였고, 자동 처리 precision은 **89.50% → 95.52%**로 개선했다.

## 핵심 결과

| 항목 | 결과 |
| --- | --- |
| 운영 후보 | KLUE-BERT (`klue/bert-base`) fine-tuning |
| 최종 test macro F1 | **0.9551** |
| 자동 처리 precision / 이관율 | **95.52% / 0.01%** |
| 데이터 | AI Hub 고객 질문 45,000건 · 3개 라벨 |
| 안전장치 | PII 마스킹 · 상담번호 그룹 분할 · validation 기반 보정 · holdout 최종 1회 평가 |

## 문제 정의와 검증 원칙

- 원본 CSV는 [데이터 계약](data/README.md)의 `text`, `label` 열을 따른다. 학습 전에 이메일·전화번호·주문번호는 `[EMAIL]`, `[PHONE]`, `[ORDER_ID]`로 치환한다.
- 라벨의 포함·제외 경계와 애매 사례는 [라벨 정의서](docs/label-guide.md)에서 관리한다. 새 업무는 기존 코드에 억지로 합치지 않고 정의서·데이터·모델 버전을 함께 갱신한다.
- `prepare_data.py`가 seed `42`의 계층화 train 70% / validation 15% / test 15% 분할을 한 번 생성한다.
- `timestamp`가 있는 데이터는 별도 날짜 기반 split을 만들어, 미래 시점의 새 표현에 대한 일반화 성능도 측정한다.
- 모델 비교·Early Stopping·자동 배정 confidence 임계값은 validation에서만 결정한다. test set은 운영 후보를 정한 뒤 한 번만 평가한다.
- 정확도 외에 macro precision/recall/F1과 클래스별 F1을 기록한다. 운영 지표는 `자동 처리율`, `자동 처리 precision`, `상담사 이관율`까지 함께 고정한다.

```text
고객 질문 → PII 마스킹 → 라벨 정규화 → 상담번호 단위 분할
→ 기준선/Transformer 학습 → validation 보정 → 자동 분류 또는 상담사 이관
```

## 재현 방법

```bash
pip install -r requirements.txt
python src/prepare_data.py

# timestamp 열이 있는 원본 데이터의 미래 시점 test 분할
python src/prepare_data.py --split-strategy temporal --output-dir data/processed-temporal

# 기준선: 형태소 분석기 없이도 재현하기 쉬운 TF-IDF + Logistic Regression
python src/train_baseline.py --run-name tfidf-logreg

# 한국어 사전학습 Transformer 미세조정
python src/train_transformer.py --run-name klue-bert-finetune
```

Transformer의 기본 모델은 `klue/bert-base`이며, 첫 실행에는 모델·토크나이저 다운로드가 필요하다. GPU를 쓸 수 있다면 `--device cuda`를 지정한다. 결과 비교 후 선택한 후보만 test set으로 평가한다.

### AI Hub 소상공인 주문 문의 데이터 적용

내려받은 AI Hub 원본이 `data/raw/Training/`, `data/raw/Validation/`에 있을 때는 일반 `prepare_data.py` 대신 아래 변환기를 사용한다. 고객 질문(`발화자=c`, `QA여부=q`)만 남기고, `배송_*`·`교환|반품|환불_*`·`제품_*` 인텐트를 각각 `delivery`·`refund`·`product`로 정규화한다. 세부 규칙은 [매핑 설정](configs/aihub-intent-map.json)에 고정한다.

```bash
python src/prepare_aihub_order_qa.py
```

현재 생성한 첫 실험 데이터는 train **30,855건**, validation **5,145건**, 공식 source validation 기반 test **9,000건**이며 세 라벨을 균형 있게 맞췄다. Training 내부는 `상담번호` 단위로 group-stratified 분할했고, source validation은 최종 test holdout으로 보존했다. train/validation·train/test의 상담번호 그룹 중복은 각각 0건이다.

원본 `날짜`는 고객 질문의 약 6%에만 있어 이 데이터에는 날짜 기반 test split을 주장하지 않는다. 날짜가 완전한 별도 문의 데이터가 생기면 그때 `prepare_data.py --split-strategy temporal` 평가를 추가한다.

```bash
# validation에서만 Temperature Scaling과 자동 처리 임계값을 고정
python src/calibration.py --run-name klue-bert-finetune --device cuda

# 기본적으로 artifacts/<run-name>/manifest.json의 보정값을 자동 적용
python src/evaluate.py --run-name klue-bert-finetune --device cuda

# 미래 시점 test 성능 확인
python src/evaluate.py --run-name klue-bert-finetune --test-csv data/processed-temporal/test.csv --device cuda
python -m unittest discover -s tests -v
```

## 실험 설계

| 실험 | 확인할 점 |
| --- | --- |
| TF-IDF + Logistic Regression | 빠르고 설명 가능한 기준 성능, 어떤 단어·n-gram이 유효한지 확인 |
| 한국어 Transformer | 문맥과 표현 변형을 반영했을 때 macro F1이 개선되는지 확인 |
| Confidence 정책 | 자동 분류 정확도를 일정 수준 이상으로 유지할 때 자동 처리 가능한 비율 확인 |

`results/validation-model-comparison.csv`로 validation 성능을 비교하고, 선택된 experiment의 최종 수치는 `results/model-comparison.csv`와 `*-test-metrics.json`에서 확인한다.

## 실험 결과

동일한 AI Hub 고객 질문 분할(train 30,855 / validation 5,145 / 공식 holdout test 9,000)에서 실행한 결과다. 모든 threshold와 Temperature Scaling은 validation에서만 고정했고, test set은 모델별 최종 1회 평가에만 사용했다.

| 지표 | TF-IDF + Logistic Regression | KLUE-BERT Fine-tuning |
| --- | ---: | ---: |
| validation macro F1 | 0.8838 | **0.9578** |
| test macro F1 | 0.8761 | **0.9551** |
| test accuracy | 0.8761 | **0.9551** |
| 자동 처리율 (test) | 95.53% | **99.99%** |
| 자동 처리 precision (test) | 89.50% | **95.52%** |
| 상담사 이관율 (test) | 4.47% | **0.01%** |
| 학습 시간 | 4.6초 | 2,569초 (약 42분 49초) |
| 모델 크기 | 9.7MB | 443.5MB |

### 해석

- KLUE-BERT는 세 라벨 모두 test F1 0.95 안팎을 달성했고, 기준선보다 macro F1을 **0.0790** 높였다.
- 반면 모델은 약 46배 크고 학습 시간도 길다. 서빙에서는 앱 시작 시 한 번만 로드하고 CPU/GPU 추론 지연·메모리를 별도로 측정해야 한다.
- 99.99%라는 높은 자동 처리율은 현재 공식 holdout과 유사한 분포에서 얻은 결과다. 채널·시기·표현이 다른 외부 문의로 정책을 재검증해야 한다.

## 운영 적용 판단

| 선택 | 근거 | 후속 확인 |
| --- | --- | --- |
| KLUE-BERT를 운영 후보로 선정 | macro F1·자동 처리 precision이 모두 가장 높음 | FastAPI 추론 지연, 메모리, 배치 처리량 측정 |
| validation threshold `0.3597` 적용 | validation에서 precision 0.9578로 목표 0.90 충족 | 신규 채널 데이터에서 자동 처리 precision 모니터링 |
| 저신뢰 예측은 상담사 이관 | test에서 1건이 이관됨 | 실제 이관 사유·상담사 재분류 결과 수집 |

## 자동 분류 / 사람 검토 정책

각 모델은 validation 예측에 Temperature Scaling을 적용한 뒤 `minimum_auto_precision`(기본 0.90)을 만족하면서 자동 처리량이 가장 큰 threshold를 선택한다. 보정 manifest가 없을 때만 학습 시 저장된 미보정 정책을 사용한다.

- `confidence >= threshold`: 자동 분류 후보
- `confidence < threshold`: `needs_review=true`, 상담사 검토

test 평가에서는 validation에서 고정한 threshold를 그대로 적용하고, 다음 운영 지표를 모두 기록한다.

| 지표 | 정의 | 운영 목적 |
| --- | --- | --- |
| macro F1 | 라벨별 F1의 단순 평균 | 소수 업무 라벨까지 놓치지 않는지 확인 |
| 자동 처리율 | 전체 중 자동 분류 건 비율 | 자동화 효율 확인 |
| 자동 처리 precision | 자동 분류 건 중 실제 정답 비율 | 잘못된 자동 배정 위험 관리 |
| 상담사 이관율 | `needs_review=true` 비율 | 상담 인력·큐 용량 계획 |

`results/generated/*-misclassifications.csv`에는 최대 120자의 마스킹된 텍스트 발췌, 정답·예측·confidence·검토 여부가 저장된다. 이 파일은 문의 텍스트를 포함할 수 있어 Git에서 제외한다.

confidence는 모델의 확률 추정치이지 사실상 보장된 신뢰도는 아니다. `results/<run-name>-calibration.json`의 보정 전후 NLL·ECE를 확인하고, 운영 전에는 시간대·채널·고객군별 성능 차이와 실제 상담사 재분류 결과를 지속적으로 점검해야 한다.

## 모듈 경계

- `src/pii.py`: 학습·평가가 함께 쓰는 최소 PII 치환 규칙
- `src/inference.py`: baseline/Transformer의 모델 로드·확률 예측만 담당한다. 평가 코드는 더 이상 Transformer 학습 모듈을 가져오지 않는다.
- `src/calibration.py`: validation 전용 Temperature Scaling과 자동 검토 policy manifest 생성
- `src/evaluate.py`: 저장된 모델과 calibration manifest를 읽어 최종 test 운영 지표를 생성
- `src/prepare_aihub_order_qa.py`: AI Hub 원본 CSV를 고객 질문·라벨 매핑·PII 마스킹·상담번호 그룹 분할 규칙에 따라 학습 CSV로 변환
