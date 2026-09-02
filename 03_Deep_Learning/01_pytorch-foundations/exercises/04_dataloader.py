"""TODO: TensorDataset과 DataLoader의 배치·shuffle 동작을 확인한다."""

import torch
from torch.utils.data import DataLoader, TensorDataset


def main() -> None:
    dataset = TensorDataset(torch.arange(11), torch.arange(11) % 2)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(42))
    for batch_index, (features, labels) in enumerate(loader, start=1):
        print(f"batch={batch_index}, features={features.tolist()}, labels={labels.tolist()}")
    # TODO: shuffle=False 결과와 비교하고 마지막 배치 크기를 설명한다.


if __name__ == "__main__":
    main()
