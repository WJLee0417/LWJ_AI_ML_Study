FROM python:3.11-slim

WORKDIR /service
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

ENV MODEL_PATH=/models/best_resnet-finetune-aug.pt
ENV MODEL_DEVICE=cpu
ENV REVIEW_THRESHOLD=0.80

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
