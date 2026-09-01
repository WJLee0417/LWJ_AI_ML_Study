# MNIST FCN vs CNN 벤치마크

MNIST 손글씨 숫자 분류에서 완전연결신경망(FCN)과 합성곱신경망(CNN)을 같은 데이터 분할과 학습 조건에서 비교한다. 기존 [FCN vs CNN 실습 노트북](../CNN/RCN_CNN_Compare.ipynb)의 설명형 코드를 재현 가능한 프로젝트 구조로 확장했다.

## 문제와 검증 원칙

- 원래 학습 데이터 60,000개를 seed `42`의 계층화 분할로 train 48,000개와 validation 12,000개로 나눈다.
- test 10,000개는 체크포인트 선택과 학습에 사용하지 않고, 학습 완료 후 `evaluate.py`에서만 평가한다.
- Python, NumPy, PyTorch 및 CUDA cuDNN의 시드를 고정한다. GPU 연산에는 환경에 따라 미세한 차이가 남을 수 있으므로 결과 실행 환경도 함께 기록한다.
- 최선의 validation accuracy를 낸 모델 하나만 `models/best_fcn.pt` 또는 `models/best_cnn.pt`로 저장한다.

## 실행

프로젝트 디렉터리에서 의존성을 설치한 뒤 실행한다.

```bash
pip install -r requirements.txt
python src/train.py --model both --epochs 5
python src/evaluate.py --model both
python -m unittest discover -s tests -v
```

GPU를 명시하려면 `--device cuda`를, CPU 재현 실행에는 `--device cpu`를 사용한다. 실행 중 다운로드되는 MNIST 데이터와 모델 가중치는 Git에 포함하지 않는다.

## 생성 결과

| 위치 | 내용 |
| --- | --- |
| `models/best_*.pt` | 최고 validation accuracy 체크포인트 |
| `results/validation-model-comparison.csv` | validation 기준 정확도·파라미터 수·학습 시간·파일 크기 |
| `results/model-comparison.csv` | 최종 test accuracy, macro precision/recall/F1, 파라미터 수·파일 크기 |
| `results/metrics.json` | 클래스별 precision, recall, F1을 포함한 상세 평가 지표 |
| `assets/loss-curve.png` | train/validation loss 곡선 |
| `assets/confusion-matrix.png` | 모델별 test confusion matrix |
| `assets/misclassified-samples.png` | 모델별 첫 오분류 test 샘플 |

## 해석 프레임

CNN의 우수성을 정확도 하나로만 판단하지 않는다. FCN은 28×28 이미지를 784개 숫자로 펼쳐 픽셀의 이웃 관계를 잃는다. 반면 CNN은 작은 필터를 공유해 획·모서리 같은 지역 패턴을 찾고, 그 패턴이 위치를 조금 옮겨도 활용한다.

최종 보고서에서는 다음을 함께 해석한다.

1. **공간 구조 활용:** 혼동행렬과 오분류 샘플에서 CNN이 줄인 오류 유형을 확인한다.
2. **파라미터 효율:** `model-comparison.csv`의 trainable parameter 수와 모델 파일 크기를 비교한다.
3. **학습·추론 비용:** `validation-model-comparison.csv`의 학습 시간을 비교하고, 실제 서비스 요구 지연 시간에서는 별도 배치 추론 측정을 추가한다.
4. **적용 적합성:** 이미지처럼 공간 패턴이 중요한 데이터에는 CNN이 적합하지만, 표 형태 데이터에는 무조건 CNN을 적용하지 않는다.
