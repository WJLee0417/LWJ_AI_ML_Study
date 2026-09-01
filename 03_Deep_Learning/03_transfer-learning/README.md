# 폐기물 이미지 분류: Scratch CNN vs ResNet18 전이학습

이미지 한 장을 `paper`, `plastic`, `metal`처럼 사용자가 정의한 폐기물 클래스 중 하나로 분류해, 분리배출 보조 API의 모델 후보를 비교한다. 작은 데이터셋에서 처음부터 CNN을 학습하는 방식과 ImageNet 사전학습 ResNet18을 비교하는 프로젝트다.

## 검증 원칙

- `data/raw/<class>/`에 넣은 원본 이미지를 클래스별로 seed `42`로 나눈다. 기본 비율은 train 70%, validation 15%, test 15%다.
- validation macro F1로 체크포인트와 Early Stopping을 결정하며, test set은 선택이 끝난 뒤 `evaluate.py`에서 한 번만 사용한다.
- 불균형 클래스에는 train 데이터의 빈도 역수 기반 class weight를 CrossEntropyLoss에 적용한다.
- Python·NumPy·PyTorch·CUDA cuDNN 시드를 고정한다. GPU 환경에 따라 아주 작은 수치 차이는 생길 수 있으므로 실행 환경도 결과에 기록한다.

## 데이터 준비

공개 데이터셋 또는 직접 수집한 이미지를 다음 구조로 배치한다. 클래스 이름은 API 응답에도 그대로 쓰이므로 영문 소문자와 하이픈처럼 안정적인 이름을 권장한다.

```text
data/raw/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

클래스마다 최소 10장 이상이 필요하며, `jpg`, `jpeg`, `png`, `webp` 파일을 지원한다. 원본 데이터와 복사된 분할 데이터는 Git에 포함하지 않는다.

```bash
pip install -r requirements.txt
python src/prepare_data.py
```

`data/processed`가 비어 있지 않으면 기존 분할을 보호하기 위해 중단한다. 새 분할이 필요하면 다른 `--output-dir`를 지정해 생성하고, 검토한 뒤 해당 경로를 학습에 전달한다.

## 핵심 실험

동일한 데이터 분할, seed, epoch 한도에서 아래 실험을 수행한다. Scratch 모델은 첫 기준선이고, ResNet18의 사전학습 가중치는 첫 실행 시 torchvision이 내려받는다.

| 실험 | 목적 | 실행 예시 |
| --- | --- | --- |
| `scratch-aug` | 작은 데이터에서 CNN 기준선 확인 | `python src/train.py --model scratch --run-name scratch-aug` |
| `resnet-feature-aug` | 사전학습 특징 추출 효과 확인 | `python src/train.py --model resnet-feature --run-name resnet-feature-aug` |
| `resnet-finetune-aug` | 전체 계층 미세조정 효과 확인 | `python src/train.py --model resnet-finetune --run-name resnet-finetune-aug --learning-rate 0.0001` |
| `resnet-finetune-noaug` | 증강의 일반화 효과 확인 | `python src/train.py --model resnet-finetune --run-name resnet-finetune-noaug --augmentation off --learning-rate 0.0001` |

필요하면 `--class-weighting none`을 별도 실험으로 실행해 불균형 보정의 영향을 비교한다. 작은 데이터에서는 validation macro F1의 평균과 클래스별 F1도 함께 보고, 단일 실행의 미세한 차이만으로 결론 내리지 않는다.

## 최종 평가

validation 결과를 비교해 하나의 운영 후보를 고른 후에만 해당 experiment를 test set으로 평가한다.

```bash
python src/evaluate.py --run-name resnet-finetune-aug
python -m unittest discover -s tests -v
```

## 결과물과 해석

| 결과물 | 의미 |
| --- | --- |
| `results/validation-model-comparison.csv` | 실험별 validation macro F1, 파라미터 수, 학습 시간, 체크포인트 크기 |
| `results/model-comparison.csv` | 선택된 실험의 최종 test accuracy 및 macro precision/recall/F1 |
| `results/*-test-metrics.json` | 클래스별 precision, recall, F1 및 support |
| `assets/*-learning-curve.png` | 증강과 정규화가 과적합에 미친 영향 확인 |
| `assets/*-confusion-matrix.png` | 서로 자주 혼동하는 재질 확인 |
| `assets/*-gradcam-errors.png` | 오분류에서 모델이 주목한 이미지 영역 확인 |

운영 후보는 정확도만으로 고르지 않는다. 소수 클래스를 놓치지 않는 macro F1, 필요한 메모리·학습 시간·추론 지연 시간, 그리고 Grad-CAM이 배경이나 워터마크가 아닌 재질 특성에 주목하는지를 함께 판단한다. Grad-CAM은 설명 보조 수단이며 인과적 근거는 아니다.
