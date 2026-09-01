import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train import load_and_split


def test_split_and_target_ratio():
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_and_split()
    assert len(x_train) + len(x_valid) + len(x_test) == len(y_train) + len(y_valid) + len(y_test)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
