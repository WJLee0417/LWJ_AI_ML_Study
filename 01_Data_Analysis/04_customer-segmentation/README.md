# 이커머스 고객 세분화와 프로모션 전략

## 문제 정의

제한된 마케팅 예산에서 모든 고객에게 같은 혜택을 제공하는 대신, 구매 행동이 유사한 고객군을 식별하고 고객군별 프로모션 전략을 설계한다.

## 데이터와 범위

- 입력 데이터: [03_sql-analysis](../03_sql-analysis/) 프로젝트가 변환한 Olist 이커머스 주문 데이터
- 분석 대상: 배송 완료 주문이 1건 이상인 고객 고유 ID
- 분석 단위: 고객 한 명당 한 행
- 제외 범위: 결제·리뷰·배송 지연은 이번 세분화 특성에 포함하지 않는다.

기존 [Cluster_ShoppingMall.ipynb](../Clustering/Cluster_ShoppingMall.ipynb)는 make_blobs로 생성한 합성 데이터 실습이다. 이 프로젝트는 해당 노트북을 삭제하거나 실제 데이터처럼 표현하지 않고, 실제 거래 데이터를 사용해 별도로 재구성한다.

## 고객 특성

| 특성 | 정의 | 역할 |
| --- | --- | --- |
| purchase_frequency | 배송 완료 주문 수 | 구매 반복성 |
| total_revenue | 상품가와 배송비의 누적 합계 | 고객 가치 |
| recency_days | 데이터 기준일과 마지막 구매일의 차이 | 이탈 위험 신호 |
| avg_order_value | 주문당 평균 매출 | 구매 단가 |
| preferred_category | 매출이 가장 큰 상품 카테고리 | 군집 해석·프로모션 소재 |

카테고리 선호도는 명목형 변수이므로 거리 기반 K-Means의 입력값에는 넣지 않는다. 군집을 만든 뒤 해석에만 사용한다.

## 방법

1. 주문·주문상품·상품·고객 테이블에서 고객 특성을 생성한다.
2. 분포가 긴 매출·주문 수에는 log1p 변환을 적용한다.
3. 모든 수치 특성에 StandardScaler를 적용한다. K-Means는 유클리드 거리 기반이므로 금액과 일수·건수를 그대로 비교하면 큰 단위 변수가 거리를 지배한다.
4. K=2~6에 대해 inertia와 silhouette score를 모두 비교한다.
5. silhouette score가 가장 높은 K를 우선 선택하되, 최소 군집 비율이 5% 미만이면 후보에서 제외한다.
6. random seed 10개와 80% 표본 반복에서 Adjusted Rand Index를 비교해 안정성을 확인한다.

## 실행 순서

먼저 3단계의 변환 데이터를 만들어야 한다.

~~~powershell
Set-Location ..\03_sql-analysis
python src/download_data.py
python src/transform_data.py

Set-Location ..\04_customer-segmentation
pip install -r requirements.txt
python src/build_features.py
python src/segment_customers.py
~~~

## 생성 결과

| 결과물 | 설명 |
| --- | --- |
| customer_features.csv | 고객별 RFM·객단가·선호 카테고리 |
| customer_segments.csv | 군집 번호와 프로필용 특성 |
| elbow-score.png | K별 inertia |
| silhouette-score.png | K별 silhouette score |
| customer-segments-pca.png | PCA 2차원 군집 분포 |
| cluster-profile.png | 군집별 표준화 특성 비교 |
| segmentation-report.md | 선택한 K, 안정성, 전략, 한계 |

## 한계

- 군집은 고객 행동의 유사성이지 인과관계나 고객의 본질적인 등급을 뜻하지 않는다.
- Olist 데이터는 하나의 기간에 국한되어 있으며 미래 행동을 보장하지 않는다.
- 할인 반응·웹 행동 로그·캠페인 전환 데이터가 없어 프로모션 효과를 직접 검증할 수 없다.
- silhouette score가 높더라도 실제 마케팅에서 행동 가능한 군집인지 별도 판단이 필요하다.
