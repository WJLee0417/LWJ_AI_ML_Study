"""Scratch CNN and ResNet18 transfer-learning model factories."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ScratchCNN(nn.Module):
    """Small baseline model for comparison with ImageNet-pretrained ResNet18."""

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


def build_model(name: str, class_count: int, pretrained: bool = True) -> nn.Module:
    """Build a baseline, frozen ResNet18 head, or fully fine-tuned ResNet18."""
    if name == "scratch":
        return ScratchCNN(class_count)
    if name not in {"resnet-feature", "resnet-finetune"}:
        raise ValueError("model must be scratch, resnet-feature, or resnet-finetune.")
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if name == "resnet-feature":
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def gradcam_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Choose the final convolutional activation source for Grad-CAM."""
    if architecture == "scratch":
        return model.features[7]
    return model.layer4[-1].conv2
