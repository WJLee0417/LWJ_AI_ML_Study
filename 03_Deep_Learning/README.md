# Deep Learning

PyTorch를 중심으로 딥러닝의 기본 구현부터 이미지·텍스트 분류 실험, 확률 보정, HTTP 추론 API 연결까지 단계적으로 다루는 학습·실험 공간이다. 각 폴더는 앞선 단계의 산출물과 판단 기준을 다음 단계에서 재사용하도록 구성했다.

이 문서는 전체 구조와 현재 확인된 결과를 요약한다. 각 실험의 세부 설정, 원본 수치, 실행 명령은 하위 프로젝트 README와 `results/`의 CSV·JSON을 기준으로 확인한다. 데이터·학습 가중치처럼 용량이 크거나 라이선스 제약이 있는 파일은 Git에 넣지 않는다.

## 한눈에 보기

| 영역 | 주제 | 현재 확인 범위 | 다음 연결 |
| --- | --- | --- | --- |
| [01 PyTorch Foundations](01_pytorch-foundations/README.md) | 텐서, 자동 미분, 학습 루프, CNN의 공간 구조 | 노트북 기반 개념 확인과 짧은 구현 과제 | MNIST 실험의 `DataLoader`, optimizer, shape 검증 |
| [02 MNIST CNN Benchmark](02_mnist-cnn-benchmark/README.md) | FCN과 CNN의 재현 가능한 비교 | CPU 5 epoch 실행, test·오분류·추론 측정 완료 | 이미지 모델의 실험 분리·평가 방식 |
| [03 Transfer Learning](03_transfer-learning/README.md) | RealWaste 이미지 분류와 ResNet18 미세조정 | RTX 4050 GPU의 최종 test·보정·Grad-CAM 생성 완료 | 이미지 추론 API와 모델 manifest |
| [04 Text Classification](04_text-classification/README.md) | 한국어 고객 문의 자동 분류 | AI Hub holdout 9,000건에서 TF-IDF와 KLUE-BERT 비교 완료 | confidence 기반 자동 분류·상담사 이관 정책 |
| [05 Model Serving](05_model-serving/README.md) | FastAPI 이미지 추론 API와 Spring 연동 계약 | 단일·배치 API, 입력 검증, 모델 manifest 검증, 테스트 구현 | 실제 배포 환경의 지연 시간·모니터링 측정 |

## 전체 구조

```text
03_Deep_Learning/
├── 01_pytorch-foundations/       # 텐서·autograd·학습 루프·CNN 기초
├── 02_mnist-cnn-benchmark/       # FCN vs CNN 재현성 벤치마크
├── 03_transfer-learning/         # RealWaste 이미지 분류 전이학습
├── 04_text-classification/       # AI Hub 한국어 문의 분류
├── 05_model-serving/             # FastAPI 추론 API 및 Spring 연동 계약
└── README.md                      # 전체 목적, 결과, 연결 관계
```

각 프로젝트 내부의 공통 역할은 다음과 같다.

| 경로 | 역할 | Git 관리 원칙 |
| --- | --- | --- |
| `src/` | 데이터 준비, 학습, 평가, 보정 등 실행 코드 | 포함 |
| `configs/` | seed, batch size, epoch, 모델·증강 설정 | 포함 |
| `tests/` | shape, 데이터 분할, API 입력 등 빠른 검증 | 포함 |
| `results/` | CSV·JSON 형태의 수치 근거 | 포함하되 텍스트 원문은 제외 |
| `assets/` | 학습 곡선, 혼동행렬, 오분류 시각화 | 포함 |
| `artifacts/<run-name>/manifest.json` | 모델과 전처리·임계값의 배포 계약 | 포함 |
| `models/` | checkpoint·tokenizer 등 대용량 모델 파일 | 제외 |
| `data/raw/`, `data/processed/` | 원본 데이터와 분할 데이터 | 제외 |

## 설계 원칙

### 1. 학습·선택·최종 평가는 분리한다

모든 비교 프로젝트는 train, validation, test의 역할을 나눈다. 학습은 train에서 수행하고, 모델·epoch·confidence threshold의 선택은 validation에서만 한다. test set은 선택이 끝난 뒤 최종 성능 확인에만 사용한다. 따라서 test 수치를 보고 설정을 다시 바꾸는 방식은 허용하지 않는다.

### 2. 같은 개체·대화가 데이터 분할을 넘지 않게 한다

이미지는 같은 물체를 연속 촬영한 경우 `group_id`로 묶어 분할할 수 있다. 고객 문의는 같은 `상담번호`의 발화가 train과 test에 함께 들어가지 않도록 그룹 단위 분할을 사용한다. 이는 유사 샘플이 섞여 성능이 과대평가되는 누수를 줄이기 위한 기준이다.

### 3. 점수 하나가 아닌 근거 묶음으로 판단한다

분류 성능은 accuracy만 기록하지 않는다. macro precision·recall·F1, 클래스별 F1, 혼동행렬, 오분류 샘플을 함께 본다. 모델 선택 시에는 파라미터 수, checkpoint 크기, 학습 시간, 추론 지연 시간도 확인한다.

### 4. 확률값은 검증한 뒤 운영 정책에 사용한다

softmax confidence를 그대로 확정 근거로 사용하지 않는다. validation logits로 Temperature Scaling을 적합하고, NLL·ECE를 통해 보정 전후를 확인한다. 그 뒤 목표 자동 처리 precision을 만족하는 threshold를 고정해 `needs_review` 정책에 사용한다.

### 5. 학습과 서빙의 계약을 파일로 남긴다

이미지 분류 운영 후보는 manifest에 모델 버전, 아키텍처, 클래스 순서, 이미지 크기, 정규화, checkpoint SHA-256, Temperature, review threshold를 기록한다. API가 이 manifest를 읽어 검증함으로써 학습 때의 클래스 순서·전처리와 배포 시의 처리가 어긋나는 위험을 줄인다.

```text
원본 데이터
  → 개인정보·누수 점검
  → train / validation / test 분할
  → 기준선과 딥러닝 모델 학습
  → validation 기반 모델 선택·확률 보정
  → holdout test 최종 평가
  → manifest 생성
  → FastAPI 예측·needs_review 반환
  → Spring BFF 또는 운영 검토 큐 연동
```

## 프로젝트별 내용과 결과

### 01. PyTorch Foundations

기본 개념을 작은 코드 단위로 확인하는 출발점이다. 수동 SGD와 역전파를 다루는 신경망 fitting, 다중 입력·출력의 shape, FCN과 CNN의 차이를 노트북으로 정리했다. `exercises/`에는 autograd, optimizer, overfitting, DataLoader를 직접 수정하거나 구현하는 짧은 과제가 있다.

| 확인 항목 | 다음 프로젝트에서의 사용 |
| --- | --- |
| Tensor shape와 행렬 연산 | 배치 입력·출력 shape 검증 |
| autograd와 `loss.backward()` | 학습 루프와 gradient update 이해 |
| optimizer와 learning rate | MNIST·전이학습의 설정 파일화 |
| overfitting과 validation | checkpoint 선택과 Early Stopping |
| 합성곱의 지역 연결·가중치 공유 | FCN 대비 CNN의 파라미터 효율 해석 |

실행 결과를 경쟁시키는 프로젝트가 아니라, 이후 실험 코드가 왜 train/eval 모드·DataLoader·validation을 필요로 하는지 이해하기 위한 단계다.

### 02. MNIST CNN Benchmark

MNIST 손글씨 숫자에서 FCN과 CNN을 같은 데이터 분할과 학습 조건으로 비교한다. 기존 설명형 노트북을 `data.py`, `models.py`, `train.py`, `evaluate.py`, `benchmark.py`로 분리하고, 단위 테스트와 smoke test를 추가했다.

2026-09-02 CPU 환경(PyTorch `2.13.0+cpu`, seed 42, 5 epoch, batch size 128)에서 확인한 결과는 아래와 같다.

| 모델 | Test accuracy | Macro F1 | 파라미터 수 | 학습 시간 | checkpoint 크기 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FCN | 0.9713 | 0.9711 | 235,146 | 87.91초 | 943,841 B |
| CNN | **0.9869** | **0.9869** | **206,922** | **86.25초** | **831,643 B** |

CPU 추론 측정에서는 FCN이 batch 1 기준 0.241 ms, CNN이 0.524 ms였다. batch 128에서는 FCN이 더 높은 throughput을 보였다. 그러나 숫자 이미지에서는 CNN이 지역 패턴을 활용해 더 높은 F1을 보이면서도 이 구성에서는 파라미터·파일 크기가 더 작았다. 따라서 “CNN이 항상 더 빠르다”가 아니라, 정확도·모델 구조·배치 조건·하드웨어를 함께 봐야 한다는 점을 확인하는 실험이다.

재실행 예시:

```bash
cd 02_mnist-cnn-benchmark
pip install -r requirements.txt
python src/train.py --model both --epochs 5
python src/evaluate.py --model both
python src/benchmark.py --device cpu --batch-sizes 1,128
python -m unittest discover -s tests -v
```

학습 곡선·혼동행렬·오분류 이미지는 [프로젝트 README](02_mnist-cnn-benchmark/README.md)와 `assets/`에서 확인할 수 있다. GPU 수치는 아직 같은 조건으로 별도 기록하지 않았으므로 CPU 수치와 직접 비교해 해석하지 않는다.

### 03. Transfer Learning

RealWaste 9개 클래스 이미지에서 Scratch CNN, ResNet18 feature extractor, ResNet18 fine-tuning, augmentation 유무를 비교할 수 있도록 구성했다. 현재 최종 후보는 ImageNet 사전학습 ResNet18 전체 계층을 미세조정하고 데이터 증강·class weight를 적용한 `resnet-finetune-aug`이다.

RTX 4050 Laptop GPU에서 수행한 최종 평가 결과다.

| 항목 | 결과 |
| --- | ---: |
| 데이터 분할 | train 3,323 / validation 712 / test 717 |
| 클래스 수 | 9 |
| 최고 validation macro F1 | 0.9299 (epoch 19) |
| test accuracy | 0.9233 |
| test macro F1 | **0.9297** |
| 파라미터 수 | 11,181,129 |
| checkpoint 크기 | 44.8 MB |
| 학습 시간 | 1,420.9초 (약 23분 41초) |

validation에만 Temperature Scaling을 적용한 뒤 NLL은 0.2542에서 0.2344로, ECE는 0.0389에서 0.0188로 낮아졌다. 자동 분류 기준 threshold는 0.2977이며 validation에서 712건 모두 자동 처리되고 precision 0.9298을 기록했다. 이 threshold는 현재 데이터와 유사한 조건에서의 검증값일 뿐이므로, 조명·배경·촬영 기기가 다른 외부 이미지에서는 다시 평가해야 한다.

test 클래스별 F1은 `miscellaneous-trash` 0.8671, `metal` 0.8843, `plastic` 0.8929가 상대적으로 낮다. 향후 데이터 수집과 오분류 검토는 이 클래스부터 우선한다. Grad-CAM은 판단 근거를 보조적으로 살피는 도구이며, 모델이 무엇을 인과적으로 학습했다는 증명으로 사용하지 않는다.

```bash
cd 03_transfer-learning
pip install -r requirements.txt
python src/prepare_data.py
python src/train.py --config configs/resnet-finetune-aug.yaml --device cuda
python src/calibration.py --run-name resnet-finetune-aug --model-version resnet18-v1
python src/evaluate.py --run-name resnet-finetune-aug --device cuda
python -m unittest discover -s tests -v
```

데이터 출처·라이선스·제외 기준·한계는 [Data Card](03_transfer-learning/docs/data-card.md), 결과 근거는 [실험 README](03_transfer-learning/README.md)의 JSON·CSV 링크와 `assets/`에서 확인한다.

### 04. Text Classification

AI Hub 소상공인 주문 질의-응답 원본에서 고객 질문만 추출해 `delivery`, `refund`, `product` 3개 업무 라벨로 분류한다. 점원·상담 답변은 입력에서 제외하며, 이메일·전화번호·주문번호는 학습 전에 치환한다. 라벨 포함·제외 기준은 [라벨 정의서](04_text-classification/docs/label-guide.md), AI Hub 세부 인텐트 매핑은 [설정 파일](04_text-classification/configs/aihub-intent-map.json)에 고정했다.

첫 실험 데이터는 train 30,855건, validation 5,145건, 공식 source validation 기반 holdout test 9,000건이다. Training 내부는 상담번호 단위 group-stratified 분할을 했고, train/validation과 train/test 사이 상담번호 중복은 각각 0건이다. 원본 날짜는 고객 질문의 약 6%에만 있어 이 데이터에 날짜 기반 일반화 성능을 주장하지 않는다.

| 지표 | TF-IDF + Logistic Regression | KLUE-BERT fine-tuning |
| --- | ---: | ---: |
| validation macro F1 | 0.8838 | **0.9578** |
| holdout test macro F1 | 0.8761 | **0.9551** |
| holdout test accuracy | 0.8761 | **0.9551** |
| 자동 처리율 | 95.53% | **99.99%** |
| 자동 처리 precision | 89.50% | **95.52%** |
| 상담사 이관율 | 4.47% | **0.01%** |
| 학습 시간 | 4.6초 | 2,569초 (약 42분 49초) |
| 모델 크기 | 9.7 MB | 443.5 MB |

KLUE-BERT의 test macro F1은 기준선보다 0.0790 높았지만, 모델 크기와 학습 시간이 크게 증가했다. 따라서 운영 후보로는 KLUE-BERT를 두되, FastAPI 배포 전 CPU/GPU 지연 시간·메모리·동시 요청 처리량을 측정해야 한다. validation에서 찾은 threshold 0.3597은 validation precision 0.9578을 만족했고, holdout test에서는 1건이 검토 대상으로 분류됐다. 다른 채널·시기·표현의 문의에서는 이 결과를 재검증해야 한다.

```bash
cd 04_text-classification
pip install -r requirements.txt
python src/prepare_aihub_order_qa.py
python src/train_baseline.py --run-name tfidf-logreg-aihub
python src/train_transformer.py --run-name klue-bert-aihub-v2 --device cuda
python src/calibration.py --run-name klue-bert-aihub-v2 --device cuda
python src/evaluate.py --run-name klue-bert-aihub-v2 --device cuda
python -m unittest discover -s tests -v
```

실제 문의 텍스트를 포함할 수 있는 오분류 CSV와 원본 데이터는 Git에서 제외한다. 자동 분류·이관 정책, calibration 지표, 결과 파일의 의미는 [프로젝트 README](04_text-classification/README.md)를 따른다.

### 05. Model Serving

전이학습의 이미지 모델을 FastAPI로 노출한다. 모델과 전처리기는 앱 시작 시 한 번만 읽고, 요청마다 checkpoint를 다시 열지 않는다. `MODEL_MANIFEST_PATH`를 지정하면 checkpoint SHA-256, 아키텍처, 클래스 순서, normalization, calibration temperature, review threshold를 검증·적용한다.

| Endpoint | 입력 | 응답·역할 |
| --- | --- | --- |
| `GET /health` | 없음 | 모델·manifest 로딩 상태 확인. 계약이 맞지 않으면 503 |
| `GET /model-info` | 없음 | 모델 아키텍처, 클래스, version, threshold 확인 |
| `POST /predict` | 최대 5 MB 이미지 한 장 | prediction, confidence, needs_review, model_version |
| `POST /predict/batch` | 최대 10장 이미지 | 이미지별 예측 결과 |

예측 응답은 아래 형태다.

```json
{
  "prediction": "plastic",
  "confidence": 0.94,
  "needs_review": false,
  "model_version": "resnet18-v1"
}
```

`needs_review=true`인 결과는 Spring BFF 또는 운영 검토 큐에서 자동 확정하지 않는 것이 계약의 핵심이다. API 테스트는 모델 stub을 주입해 정상 요청, 비이미지·손상 파일·크기 초과 입력, 단일·배치 예측을 검사한다. 실제 checkpoint를 넣은 환경의 end-to-end 지연 시간과 부하 측정은 별도로 남겨야 하며, 현재 문서는 그러한 운영 성능 수치를 주장하지 않는다.

```bash
cd 05_model-serving
pip install -r requirements.txt
uvicorn app.main:app --reload
python -m unittest discover -s tests -v
```

환경 변수, Docker 실행, Spring의 역할과 상태 코드 변환은 [서빙 README](05_model-serving/README.md)와 [Spring 연동 계약](05_model-serving/docs/spring-integration.md)에 정리했다.

## 재현성과 결과물 확인

### 실행 환경을 함께 남긴다

실험마다 seed, 라이브러리 버전, device, CUDA 사용 가능 여부, batch size, epoch, 학습 시간을 결과 파일에 기록한다. GPU의 완전한 비트 단위 동일성은 드라이버·cuDNN·하드웨어 차이로 보장되지 않을 수 있으므로, 수치 차이가 있을 때는 환경 정보를 먼저 비교한다.

### 결과 파일의 우선순위

| 확인하려는 내용 | 우선 파일 |
| --- | --- |
| 최종 test accuracy·macro F1 | 각 프로젝트 `results/model-comparison.csv`, `*-test-metrics.json` |
| validation 기준 모델 선택 근거 | `results/validation-model-comparison.csv` |
| 클래스별 오류 | `metrics.json`, `*-test-metrics.json`, confusion matrix |
| 학습 과정 | learning curve, validation 결과 CSV |
| 자동 처리·상담사 이관 정책 | calibration JSON, `artifacts/<run-name>/manifest.json` |
| 서빙 모델과 전처리의 일치 | manifest와 `05_model-serving`의 `/model-info` |

### 검증 명령

각 프로젝트의 단위 테스트는 해당 폴더에서 실행한다.

```bash
python -m unittest discover -s tests -v
```

MNIST에는 GitHub Actions smoke test가 있어 MNIST 전체 다운로드 없이 두 모델의 한 optimizer step을 확인한다. 이미지·텍스트 모델의 전체 학습은 데이터와 GPU 유무에 따라 비용이 크므로 CI에서 무조건 재실행하지 않으며, 고정된 결과 CSV·JSON·manifest와 테스트를 함께 검토한다.

## 데이터와 모델 파일 관리

- 원본 데이터는 각 프로젝트의 `data/raw/`에 두고 Git에 올리지 않는다.
- 생성된 분할 데이터도 보통 `data/processed/`에 두고 제외한다. 분할 규칙과 요약 통계는 코드·README·작은 manifest로 남긴다.
- `.pt`, `.bin`, tokenizer 파일 등 모델 가중치는 `models/`에 두고 제외한다.
- 재현에 필요한 설정·라벨 매핑·데이터 카드·결과 지표·시각화·model manifest는 Git에 포함한다.
- 문의 텍스트처럼 민감한 내용이 남을 수 있는 오분류 표본은 PII를 치환하더라도 기본적으로 Git에 올리지 않는다.

## 다음 정비 우선순위

1. `05_model-serving`에 RealWaste 운영 checkpoint를 연결해 `/health`, `/model-info`, `/predict`, `/predict/batch`의 실제 end-to-end 동작과 지연 시간을 측정한다.
2. 이미지·텍스트 모델 모두 외부 환경 또는 미래 시점 데이터로 재평가해 현재 holdout과 다른 분포에서의 성능·자동 처리 precision을 확인한다.
3. `03_transfer-learning`은 취약 클래스(`miscellaneous-trash`, `metal`, `plastic`)의 오분류를 검토하고, 중복 촬영 방지 manifest를 갖춘 추가 데이터를 수집한다.
4. `04_text-classification`은 날짜가 완전한 문의 데이터가 확보되면 상담번호 그룹을 유지한 temporal split을 추가하고, 상담사 재분류 결과를 운영 지표에 반영한다.
5. 추론 지연 시간, 메모리, 오류율, review rate를 정기적으로 기록할 수 있는 관측 방식을 정한다. 모델 성능이 높아도 입력 분포와 운영 비용이 바뀌면 정책은 다시 검증해야 한다.
