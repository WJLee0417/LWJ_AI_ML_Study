# 대출 연체 위험 예측 API

신청자 정보를 바탕으로 다음 달 연체 확률을 추정하고, 심사 담당자가 검토 우선순위를 정하는 데 쓰는 보조 의견을 제공한다. 이 서비스는 자동 승인·거절 시스템이 아니다.

## 문제와 데이터

| 항목 | 내용 |
| --- | --- |
| 목표 | 연체 위험 신청자를 우선 검토해 심사 자원을 배분한다. |
| 타깃 | default_next_month: 다음 달 연체 여부 |
| 원본 | UCI Default of Credit Card Clients, 30,000건 |
| 제외 변수 | ID: 식별자이므로 모델 입력에서 제외 |
| 데이터 범위 | 특정 국가·시기의 신용카드 고객 표본으로, 다른 대출 상품에 일반화할 수 없다. |

원본 Excel 파일은 data/raw에 두고, prepare_data.py로 정제 CSV를 재생성한다. 원본·정제 데이터·모델 파일은 Git에 커밋하지 않는다.

## 누수 방지와 평가 설계

- Train 60%에서 모델을 학습한다.
- Validation 20%에서 모델을 비교하고 임계값 정책을 선택한다.
- Test 20%는 선택된 모델과 임계값의 최종 평가에 한 번만 사용한다.
- StandardScaler와 SMOTE는 학습 파이프라인 내부에서만 적용한다.
- 예측 시점 이후 정보, 식별자 ID, 타깃 컬럼은 입력에서 제외한다.

## 모델 비교

평가 기준은 불균형 데이터에 적합한 Validation PR-AUC이며, 보조 지표로 Recall과 Precision을 함께 확인했다.

| 모델 | Validation PR-AUC | Recall | Precision | 선택 |
| --- | ---: | ---: | ---: | --- |
| Random Forest | 0.530 | 0.569 | 0.496 | 선택 |
| 기본 Logistic Regression | 0.478 | 0.226 | 0.661 | 비교 |
| SMOTE Logistic Regression | 0.478 | 0.615 | 0.363 | 비교 |
| class-weight Logistic Regression | 0.476 | 0.607 | 0.373 | 비교 |

SMOTE가 항상 더 좋은 결과를 보인 것은 아니다. 이번 검증 데이터에서는 Random Forest가 가장 높은 PR-AUC를 보여 최종 후보로 선택했다.

## 임계값 정책과 최종 결과

| 임계값 | Validation 심사 대상 | Recall | Precision | 운영 의미 |
| --- | ---: | ---: | ---: | --- |
| 0.30 | 3,586명 | 0.854 | 0.316 | 연체 누락을 줄이는 기본 추가 심사안 |
| 0.50 | 1,521명 | 0.569 | 0.496 | 심사 대상 수를 줄인 균형안 |
| 0.70 | 781명 | 0.364 | 0.618 | 고위험군 집중안 |

최종 정책은 연체 누락 비용을 우선 고려해 0.30으로 선택했다. 최종 Test 결과는 PR-AUC 0.567, Recall 0.852, Precision 0.320이다. 임계값은 심사 보조 기준이며, 승인·거절을 자동으로 결정하지 않는다.

## API 계약

실행:

    uvicorn app.main:app --reload

예측 엔드포인트는 POST /predict 이다. 응답에는 아래 정보를 포함한다.

| 필드 | 의미 |
| --- | --- |
| default_probability | 모델이 계산한 연체 확률 |
| risk_grade | low, medium, high 위험 등급 |
| decision_support | 승인 보조 의견, 추가 심사, 고위험 추가 심사 |
| policy_threshold | 응답 정책에 적용한 경계값 |
| policy_reason | 해당 심사 의견이 나온 확률 구간 근거 |
| top_risk_signals | 상환 상태·청구액·상환액 기반의 참고 신호 |
| warning | 자동 승인·거절 금지 안내 |

### 입력 검증

- LIMIT_BAL은 0보다 커야 한다.
- AGE는 18부터 100까지 허용한다.
- PAY_0부터 PAY_6까지는 -2부터 9까지 허용한다.
- EDUCATION은 0부터 6, MARRIAGE는 0부터 3, SEX는 1 또는 2만 허용한다.
- PAY_AMT1부터 PAY_AMT6까지는 음수를 허용하지 않는다.

필수 필드 누락과 범위 오류는 422 응답으로 차단한다.

## 공정성 점검

최종 Test 세트에서 성별 코드와 연령대별 실제 연체율, 예측 양성률, Recall, FPR을 비교했다. 이는 배포 전 편향 모니터링이며, 차별 여부 또는 인과관계를 확정하는 평가는 아니다.

- 성별 코드별 예측 양성률은 남성 코드 0.624, 여성 코드 0.565였다. FPR은 각각 0.546, 0.494였다.
- 60세 이상 집단은 표본이 82명이고 Recall이 0.762로 다른 연령대보다 낮았다. 작은 표본의 변동을 고려해 더 큰 최신 표본에서 지속적으로 점검해야 한다.
- 성별과 연령은 감사용 집계 지표이며, 자동 승인·거절의 근거로 사용하지 않는다.

상세 결과는 results/fairness-by-group.csv와 results/fairness-report.md에 저장된다.

## 재현과 테스트

    python src/prepare_data.py
    python src/train.py
    python src/analyze_fairness.py
    $env:PYTHONPATH='.'; python tests/test_prediction_api.py
    $env:PYTHONPATH='.'; python tests/test_fairness.py

테스트는 다음 계약을 확인한다.

- 정상 요청은 200과 0부터 1 사이의 확률을 반환한다.
- 필수 입력 누락, 한도·나이·상환 상태·범주 코드 범위 오류는 422를 반환한다.
- 0.30과 0.70 경계값에서 위험 등급, 심사 의견, 정책 근거가 일치한다.
- 위험 신호는 상환 상태와 청구·상환 정보만 사용하며 민감 특성을 노출하지 않는다.
- 공정성 지표 계산은 고객 수, Recall, FPR을 정확히 집계한다.

## 한계와 다음 개선

- 원본 데이터의 국가·시기·상품 범위가 제한적이다.
- 민감 특성의 모델 사용 여부와 대체 변수의 편향을 별도 검토해야 한다.
- 임계값은 실제 심사 비용, 회수율, 고객 영향 데이터를 바탕으로 재조정해야 한다.
- 확률 보정, 시간 기준 검증, 모델 드리프트 모니터링을 추가할 수 있다.
