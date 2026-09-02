"""HTTP API for the trained waste-image classifier."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from app.model_loader import ImagePredictor, load_configured_predictor
from app.schemas import BatchPredictionResponse, HealthResponse, ModelInfoResponse, PredictionResponse

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_BATCH_SIZE = 10


class Predictor(Protocol):
    def predict(self, image: Image.Image) -> PredictionResponse: ...
    def model_info(self) -> ModelInfoResponse: ...


async def read_image(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Each image must be 5 MB or smaller.")
    try:
        with Image.open(io.BytesIO(content)) as opened:
            return opened.convert("RGB")
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.") from error


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """Create an app; injecting a predictor makes API tests independent of model files."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor
        app.state.load_error = None
        if predictor is None:
            try:
                app.state.predictor = load_configured_predictor()
            except Exception as error:  # Health endpoint must explain an unavailable model.
                app.state.load_error = str(error)
        yield

    app = FastAPI(title="Waste Classification API", version="1.0.0", lifespan=lifespan)

    def get_predictor() -> Predictor:
        loaded = app.state.predictor
        if loaded is None:
            raise HTTPException(status_code=503, detail=f"Model is unavailable: {app.state.load_error}")
        return loaded

    @app.get("/health", response_model=HealthResponse)
    def health(response: Response) -> HealthResponse:
        if app.state.predictor is None:
            response.status_code = 503
            return HealthResponse(status="unavailable", model_loaded=False, detail=app.state.load_error)
        return HealthResponse(status="ok", model_loaded=True)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        return get_predictor().model_info()

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        file: Annotated[UploadFile, File(description="A single waste image")],
        top_k: Annotated[int, Query(ge=1, le=1, description="Reserved for a stable single-label contract")] = 1,
    ) -> PredictionResponse:
        _ = top_k
        return get_predictor().predict(await read_image(file))

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(files: Annotated[list[UploadFile], File(description="One to ten waste images")]) -> BatchPredictionResponse:
        if not 1 <= len(files) <= MAX_BATCH_SIZE:
            raise HTTPException(status_code=422, detail=f"Upload between 1 and {MAX_BATCH_SIZE} images.")
        images = [await read_image(file) for file in files]
        predictor_instance = get_predictor()
        return BatchPredictionResponse(predictions=[predictor_instance.predict(image) for image in images])

    return app


app = create_app()
