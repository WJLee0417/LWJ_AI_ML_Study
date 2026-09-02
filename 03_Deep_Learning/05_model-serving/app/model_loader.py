"""Load a saved transfer-learning checkpoint once and run image inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18

from app.schemas import ModelInfoResponse, PredictionResponse

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ScratchCNN(nn.Module):
    """Architecture matching the scratch baseline in 03_transfer-learning."""

    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, class_count))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model(architecture: str, class_count: int) -> nn.Module:
    if architecture == "scratch":
        return ScratchCNN(class_count)
    if architecture in {"resnet-feature", "resnet-finetune"}:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, class_count)
        return model
    raise ValueError(f"Unsupported architecture in checkpoint: {architecture}")


def default_model_path() -> Path:
    deep_learning_dir = Path(__file__).resolve().parents[2]
    return deep_learning_dir / "03_transfer-learning" / "models" / "best_resnet-finetune-aug.pt"


@dataclass
class ImagePredictor:
    model: nn.Module
    class_names: list[str]
    architecture: str
    checkpoint_epoch: int | None
    model_version: str
    review_threshold: float
    device: torch.device

    def __post_init__(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        model_version: str | None = None,
        review_threshold: float = 0.8,
        device: str = "cpu",
    ) -> "ImagePredictor":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        if not 0 < review_threshold <= 1:
            raise ValueError("review_threshold must be in (0, 1].")
        target_device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        for key in ("architecture", "class_names", "state_dict"):
            if key not in checkpoint:
                raise ValueError(f"Checkpoint is missing '{key}'. Use a 03_transfer-learning checkpoint.")
        model = build_model(checkpoint["architecture"], len(checkpoint["class_names"])).to(target_device)
        model.load_state_dict(checkpoint["state_dict"])
        version = model_version or f"{checkpoint['architecture']}-v{checkpoint.get('epoch', 1)}"
        return cls(model, checkpoint["class_names"], checkpoint["architecture"], checkpoint.get("epoch"), version, review_threshold, target_device)

    def predict(self, image: Image.Image) -> PredictionResponse:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        confidence, class_index = torch.max(probabilities, dim=0)
        confidence_value = float(confidence.item())
        return PredictionResponse(
            prediction=self.class_names[int(class_index.item())],
            confidence=confidence_value,
            needs_review=confidence_value < self.review_threshold,
            model_version=self.model_version,
        )

    def model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_version=self.model_version,
            architecture=self.architecture,
            class_names=self.class_names,
            review_threshold=self.review_threshold,
            checkpoint_epoch=self.checkpoint_epoch,
        )


def load_configured_predictor() -> ImagePredictor:
    path = Path(os.getenv("MODEL_PATH", str(default_model_path())))
    requested_device = os.getenv("MODEL_DEVICE", "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MODEL_DEVICE=cuda was configured but CUDA is unavailable.")
    return ImagePredictor.from_checkpoint(
        path,
        model_version=os.getenv("MODEL_VERSION"),
        review_threshold=float(os.getenv("REVIEW_THRESHOLD", "0.80")),
        device=requested_device,
    )
