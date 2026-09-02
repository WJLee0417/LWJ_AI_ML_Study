import io
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_app
from app.schemas import ModelInfoResponse, PredictionResponse


class StubPredictor:
    def predict(self, _image: Image.Image) -> PredictionResponse:
        return PredictionResponse(prediction="plastic", confidence=0.94, needs_review=False, model_version="resnet18-v1")

    def model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(model_version="resnet18-v1", architecture="resnet-finetune", class_names=["glass", "plastic"], review_threshold=0.8, checkpoint_epoch=4)


def image_payload() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 16), color="green").save(output, format="PNG")
    return output.getvalue()


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client_context = TestClient(create_app(StubPredictor()))
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_health_and_model_info(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "model_loaded": True, "detail": None})
        self.assertEqual(self.client.get("/model-info").json()["model_version"], "resnet18-v1")

    def test_single_and_batch_predictions(self):
        single = self.client.post("/predict", files={"file": ("waste.png", image_payload(), "image/png")})
        self.assertEqual(single.status_code, 200)
        self.assertEqual(single.json()["prediction"], "plastic")
        batch = self.client.post("/predict/batch", files=[("files", ("one.png", image_payload(), "image/png")), ("files", ("two.png", image_payload(), "image/png"))])
        self.assertEqual(batch.status_code, 200)
        self.assertEqual(len(batch.json()["predictions"]), 2)

    def test_invalid_upload_is_rejected(self):
        response = self.client.post("/predict", files={"file": ("note.txt", b"not an image", "text/plain")})
        self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
