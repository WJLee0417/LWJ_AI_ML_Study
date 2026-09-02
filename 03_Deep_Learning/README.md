# 🧠 03. Deep Learning (딥러닝 및 인공신경망)

PyTorch와 TensorFlow를 활용하여 인공신경망의 구조를 학습하고, 더 복잡한 패턴을 인식하는 딥러닝 모델을 구현하는 공간입니다.

## 🎯 학습 목표
* **신경망 구조 이해:** 퍼셉트론(Perceptron)부터 CNN, RNN 등 다양한 딥러닝 아키텍처의 원리를 학습합니다.
* **프레임워크 활용:** PyTorch를 중심으로 텐서(Tensor) 연산과 역전파(Backpropagation) 과정을 코드로 구현합니다.
* **AI Serving 준비:** 학습이 완료된 가중치(Weight) 모델을 저장하고, 향후 Java/Spring 기반의 백엔드 API 서버에서 이 모델을 호출하여 서빙(Serving)할 수 있는 구조를 구상합니다.

## 📁 포트폴리오 로드맵

```text
03_Deep_Learning/
├── 01_pytorch-foundations/       # 텐서·autograd·학습 루프 기초
├── 02_mnist-cnn-benchmark/       # FCN vs CNN 재현성 벤치마크
├── 03_transfer-learning/         # 사전학습 모델 기반 이미지 분류
├── 04_text-classification/       # NLP 분류 모델
└── 05_model-serving/             # FastAPI 추론 API 및 Spring 연동
```

## 📝 실습 내용

- `01_pytorch-foundations/`: 기존 신경망·CNN 노트북을 정리한 텐서, autograd, 학습 루프, 이미지 표현 기초와 짧은 구현 과제
- `02_mnist-cnn-benchmark/`: 분리된 train/validation/test, 체크포인트, 테스트, 오분류 분석을 갖춘 첫 완성 프로젝트
- `03_transfer-learning/`: 폐기물 이미지에서 Scratch CNN·ResNet18 특징 추출·미세조정·증강 효과를 비교하는 전이학습 프로젝트
- `04_text-classification/`: TF-IDF 기준선과 한국어 Transformer를 비교하고 confidence 기반 상담사 검토 정책을 적용하는 문의 분류 프로젝트
- `05_model-serving/`: 전이학습 이미지 모델을 FastAPI로 제공하고 Spring BFF 연동 계약과 API 테스트를 갖춘 서빙 프로젝트
