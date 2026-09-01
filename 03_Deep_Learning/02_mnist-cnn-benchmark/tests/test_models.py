import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import CNN, FCN, count_parameters


class ModelTests(unittest.TestCase):
    def test_models_return_ten_class_logits(self):
        inputs = torch.randn(4, 1, 28, 28)
        for model in (FCN(), CNN()):
            self.assertEqual(tuple(model(inputs).shape), (4, 10))

    def test_cnn_uses_fewer_trainable_parameters_than_fcn(self):
        self.assertLess(count_parameters(CNN()), count_parameters(FCN()))


if __name__ == "__main__":
    unittest.main()
