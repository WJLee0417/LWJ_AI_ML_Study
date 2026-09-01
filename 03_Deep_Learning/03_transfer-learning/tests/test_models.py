import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import ScratchCNN, build_model, count_parameters


class ModelTests(unittest.TestCase):
    def test_scratch_cnn_returns_one_logit_per_class(self):
        self.assertEqual(tuple(ScratchCNN(4)(torch.randn(2, 3, 224, 224)).shape), (2, 4))

    def test_feature_extractor_only_trains_replacement_head(self):
        model = build_model("resnet-feature", class_count=4, pretrained=False)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.layer1.parameters()))
        self.assertGreater(count_parameters(model), 0)
        self.assertLess(count_parameters(model), sum(parameter.numel() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
