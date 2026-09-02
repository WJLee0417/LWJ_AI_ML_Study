"""Stable HTTP response contracts for model consumers."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: str | None = None


class PredictionResponse(BaseModel):
    prediction: str = Field(description="Predicted waste class code")
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    model_version: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    model_version: str
    architecture: str
    class_names: list[str]
    review_threshold: float = Field(ge=0.0, le=1.0)
    checkpoint_epoch: int | None = None
