# PyTorch 기초 과제

각 과제는 20~40분 안에 끝낼 수 있는 작은 코드 수정 과제다. 결과 숫자보다 텐서 shape과 검증 근거를 README 또는 노트에 남긴다.

| 과제 | 목표 | 완료 기준 |
| --- | --- | --- |
| `01_autograd.py` | 자동 미분과 수치 미분 비교 | 두 gradient 값이 허용 오차 안에서 일치 |
| `02_optimizer.py` | 수동 SGD와 `torch.optim.SGD` 비교 | 같은 초기값에서 loss 감소 추세가 확인됨 |
| `03_overfitting.py` | 과적합 징후와 regularization 확인 | train/validation loss 차이를 그래프로 설명 |
| `04_dataloader.py` | Dataset·DataLoader 배치 흐름 구현 | 마지막 배치와 shuffle 동작을 출력으로 검증 |

완료 후에는 같은 개념이 [02_mnist-cnn-benchmark](../../02_mnist-cnn-benchmark/README.md)에서 어떤 API와 결과물로 확장되는지 한 문장으로 정리한다.
