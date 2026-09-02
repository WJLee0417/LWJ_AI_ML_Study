"""Run one optimizer step for both architectures without downloading MNIST."""

import torch
from torch import nn, optim

from models import build_model


def main() -> None:
    inputs = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    for name in ("fcn", "cnn"):
        model = build_model(name)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss), f"{name} produced a non-finite loss"
    print("MNIST model smoke test passed.")


if __name__ == "__main__":
    main()
