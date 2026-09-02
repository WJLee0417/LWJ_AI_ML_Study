# 폐기물 이미지 분류 FastAPI 서빙

`03_transfer-learning`에서 학습한 Scratch CNN 또는 ResNet18 체크포인트를 HTTP API로 제공한다. 모델과 전처리기는 앱 시작 시 한 번만 로드하며, 요청마다 다시 파일을 읽지 않는다.

## 시작 전 준비

먼저 전이학습 프로젝트에서 체크포인트를 생성한다.

```bash
cd ../03_transfer-learning
python src/train.py --model resnet-finetune --run-name resnet-finetune-aug --learning-rate 0.0001
```

그 뒤 이 프로젝트에서 모델 경로와 운영 정책을 설정한다. `REVIEW_THRESHOLD`는 운영에서 자동 확정해도 되는 최소 confidence이며, validation 결과를 검토해 정한다.

```powershell
pip install -r requirements.txt
$env:MODEL_PATH = "C:\path\to\03_transfer-learning\models\best_resnet-finetune-aug.pt"
$env:MODEL_VERSION = "resnet18-v1"
$env:REVIEW_THRESHOLD = "0.80"
uvicorn app.main:app --reload
```

환경 변수를 생략하면 `03_transfer-learning/models/best_resnet-finetune-aug.pt`를 기본 경로로 찾는다. 체크포인트가 없거나 구조가 맞지 않으면 서버는 기동되지만 `/health`는 `503`을 반환한다.

## API 계약

| Endpoint | 역할 |
| --- | --- |
| `GET /health` | 모델 로딩 상태 확인 |
| `GET /model-info` | 아키텍처, 클래스 목록, version, 검토 임계값 확인 |
| `POST /predict` | 단일 이미지 분류 |
| `POST /predict/batch` | 최대 10장 이미지 일괄 분류 |

`/predict`는 `multipart/form-data`의 `file` 필드에 PNG/JPEG 등 이미지 파일 하나를 받는다. 각 파일은 5 MB 이하여야 하며, 빈 파일·손상된 이미지·이미지가 아닌 MIME type은 `4xx`로 거절한다.

```json
{
  "prediction": "plastic",
  "confidence": 0.94,
  "needs_review": false,
  "model_version": "resnet18-v1"
}
```

`needs_review`가 `true`면 BFF는 결과를 자동 확정하지 않고 상담사 또는 운영자 검토 큐로 보내야 한다. 구체적인 Spring 역할과 상태 코드 변환은 [Spring 연동 계약](docs/spring-integration.md)을 따른다.

## 테스트와 컨테이너 실행

```bash
python -m unittest discover -s tests -v
docker build -t waste-model-api .
docker run --rm -p 8000:8000 -v "${PWD}/../03_transfer-learning/models:/models:ro" -e MODEL_VERSION=resnet18-v1 waste-model-api
```

테스트는 실제 모델 가중치 대신 예측기 stub을 주입해 정상 응답, 유효하지 않은 입력, 단일·배치 요청을 빠르게 검증한다.
