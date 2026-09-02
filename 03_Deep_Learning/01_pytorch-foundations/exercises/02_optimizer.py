"""TODO: 같은 선형 회귀 문제를 수동 SGD와 torch.optim.SGD로 학습한다."""

import torch


def main() -> None:
    torch.manual_seed(42)
    features = torch.arange(0, 10, dtype=torch.float32).unsqueeze(1)
    targets = 2 * features + 1
    weight = torch.randn(1, 1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    # TODO: 수동 업데이트 루프를 100 epoch 작성하고 마지막 loss를 출력한다.
    # TODO: nn.Linear + torch.optim.SGD로 같은 실험을 재현하고 loss 곡선을 비교한다.
    print(features.shape, targets.shape, weight.item(), bias.item())


if __name__ == "__main__":
    main()
