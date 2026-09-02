"""TODO: autograd gradient를 중앙 차분 수치 미분과 비교한다."""

import torch


def numeric_gradient(function, value: float, epsilon: float = 1e-4) -> float:
    return (function(value + epsilon) - function(value - epsilon)) / (2 * epsilon)


def main() -> None:
    x = torch.tensor(2.0, requires_grad=True)
    loss = x**3 + 2 * x
    loss.backward()
    manual = numeric_gradient(lambda value: value**3 + 2 * value, 2.0)
    print(f"autograd={x.grad.item():.4f}, numeric={manual:.4f}")
    # TODO: abs(x.grad.item() - manual) < 1e-3인지 assertion을 추가한다.


if __name__ == "__main__":
    main()
