# Spring BFF 연동 계약

Spring 서비스는 인증·업로드 권한·도메인 규칙을 담당하고, Python API는 모델 추론만 담당한다. 브라우저가 Python 서비스에 직접 접근하게 하지 않고 Spring BFF가 호출하면 모델의 내부 경로와 교체 전략을 숨길 수 있다.

## 호출 정책

| Python 응답 | Spring 처리 |
| --- | --- |
| `200 /predict` | 분류 결과와 confidence를 클라이언트에 전달하고 도메인 이벤트 기록 |
| `422` | 지원하지 않는 비어 있거나 손상된 이미지로 사용자 입력 오류 반환 |
| `413` | 업로드 용량 제한 오류 반환 |
| `503` | 모델 준비 실패로 재시도 가능한 서비스 오류 반환 |

Spring은 `needs_review=true`이면 자동 분류 결과를 확정하지 않고, 검토 대기 상태로 저장한다. `model_version`은 예측 이력에 함께 보관해 모델 교체 뒤에도 결과를 추적할 수 있게 한다.

```java
webClient.post()
    .uri("http://waste-model:8000/predict")
    .contentType(MediaType.MULTIPART_FORM_DATA)
    .body(BodyInserters.fromMultipartData("file", imageResource))
    .retrieve()
    .bodyToMono(WastePrediction.class);
```

Python API 호출에는 짧은 connect/read timeout과 제한된 재시도를 적용한다. 이미지 업로드 재시도는 멱등한 저장 정책과 함께 설계한다.
