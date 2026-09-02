import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pii import mask_pii


class PiiMaskingTests(unittest.TestCase):
    def test_masks_email_phone_and_order_number(self):
        text = "010-1234-5678로 연락 주세요. 이메일은 hello@example.com, 주문번호 AB-123456입니다."
        masked, counts = mask_pii(text)
        self.assertIn("[PHONE]", masked)
        self.assertIn("[EMAIL]", masked)
        self.assertIn("[ORDER_ID]", masked)
        self.assertEqual(counts, {"email": 1, "phone": 1, "order_id": 1})


if __name__ == "__main__":
    unittest.main()
