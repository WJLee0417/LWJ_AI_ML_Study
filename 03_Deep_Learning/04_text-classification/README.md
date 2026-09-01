# 한국어 고객 문의 자동 분류

고객 문의 텍스트를 배송·환불·계정·상품 등 업무 라벨로 분류하고, 충분히 확신하는 건만 자동 배정하며 나머지는 상담사 검토로 넘기는 모델을 만든다. 기존 머신러닝 분류 프로젝트의 임계값 의사결정 방식을 딥러닝 분류와 연결한다.

## 데이터와 검증 원칙

- 원본 CSV는 [데이터 계약](data/README.md)의 `text`, `label` 열을 따른다.
- `prepare_data.py`가 seed `42`의 계층화 train 70% / validation 15% / test 15% 분할을 한 번 생성한다.
- 모델 비교·Early Stopping·자동 배정 confidence 임계값은 validation에서만 결정한다. test set은 운영 후보를 정한 뒤 한 번만 평가한다.
- 정확도 외에 macro precision/recall/F1과 클래스별 F1을 기록한다. 지원 건수가 적은 라벨을 놓치지 않기 위해 macro F1을 핵심 모델 선택 지표로 둔다.

## 실행

```bash
pip install -r requirements.txt
python src/prepare_data.py

# 기준선: 형태소 분석기 없이도 재현하기 쉬운 TF-IDF + Logistic Regression
python src/train_baseline.py --run-name tfidf-logreg

# 한국어 사전학습 Transformer 미세조정
python src/train_transformer.py --run-name klue-bert-finetune
```

Transformer의 기본 모델은 `klue/bert-base`이며, 첫 실행에는 모델·토크나이저 다운로드가 필요하다. GPU를 쓸 수 있다면 `--device cuda`를 지정한다. 결과 비교 후 선택한 후보만 test set으로 평가한다.

```bash
python src/evaluate.py --run-name klue-bert-finetune
python -m unittest discover -s tests -v
```

## 비교 프레임

| 실험 | 확인할 점 |
| --- | --- |
| TF-IDF + Logistic Regression | 빠르고 설명 가능한 기준 성능, 어떤 단어·n-gram이 유효한지 확인 |
| 한국어 Transformer | 문맥과 표현 변형을 반영했을 때 macro F1이 개선되는지 확인 |
| Confidence 정책 | 자동 분류 정확도를 일정 수준 이상으로 유지할 때 자동 처리 가능한 비율 확인 |

`results/validation-model-comparison.csv`로 validation 성능을 비교하고, 선택된 experiment의 최종 수치는 `results/model-comparison.csv`와 `*-test-metrics.json`에서 확인한다.

## 자동 분류 / 사람 검토 정책

각 모델은 validation 예측의 최대 클래스 확률을 사용해 `minimum_auto_precision`(기본 0.90)을 만족하면서 자동 처리량이 가장 큰 threshold를 선택한다.

- `confidence >= threshold`: 자동 분류 후보
- `confidence < threshold`: `needs_review=true`, 상담사 검토

test 평가에서는 validation에서 고정한 threshold를 그대로 적용하고, 자동 처리율과 자동 처리 건의 실제 precision을 별도로 기록한다. `results/generated/*-misclassifications.csv`에는 최대 120자의 텍스트 발췌, 정답·예측·confidence·검토 여부가 저장된다. 이 파일은 문의 텍스트를 포함할 수 있어 Git에서 제외한다.

confidence는 모델의 확률 추정치이지 사실상 보장된 신뢰도는 아니다. 운영 전에는 시간대·채널·고객군별 성능 차이와 실제 상담사 재분류 결과를 지속적으로 점검해야 한다.
