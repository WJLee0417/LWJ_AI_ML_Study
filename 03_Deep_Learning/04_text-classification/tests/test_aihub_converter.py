import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prepare_aihub_order_qa import group_stratified_split, map_intent


class AihubConverterTests(unittest.TestCase):
    def test_maps_only_documented_intent_prefixes(self):
        mapping = [
            {"label": "delivery", "intent_prefixes": ["배송_"]},
            {"label": "refund", "intent_prefixes": ["교환|반품|환불_"]},
        ]
        self.assertEqual(map_intent("배송_비용_질문", mapping), "delivery")
        self.assertEqual(map_intent("교환|반품|환불_일반_질문", mapping), "refund")
        self.assertIsNone(map_intent("제품_정보_질문", mapping))

    def test_group_split_keeps_conversations_together(self):
        frame = pd.DataFrame(
            {
                "text": [f"문의 {index}" for index in range(12)],
                "label": ["delivery", "refund"] * 6,
                "group_id": [f"g{index // 2}" for index in range(12)],
            }
        )
        train, validation = group_stratified_split(frame, 0.25, 42)
        self.assertFalse(set(train["group_id"]) & set(validation["group_id"]))
        self.assertEqual(len(train) + len(validation), len(frame))


if __name__ == "__main__":
    unittest.main()
