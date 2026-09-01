# Data Analysis Portfolio

데이터 품질 문제를 해결하고, 탐색적 분석과 SQL을 거쳐 고객 세분화 및 상권 의사결정 대시보드까지 연결한 프로젝트 기록입니다.

## Project Map

~~~text
01_Data_Analysis/
├── 01. Data Quality        Online Retail 주문 데이터 정제
├── 02. EDA                 서울 공공자전거 수요 패턴 분석
├── 03. SQL Analytics       Olist 이커머스 관계형 분석
├── 04. Customer Segment    RFM 기반 고객 세분화
└── 05. Featured            강남구 카페업 상권 분석 대시보드
~~~

| 프로젝트 | 문제 | 핵심 기술 | 결과 | 링크 |
| --- | --- | --- | --- | --- |
| 01. Data Quality | 결측·중복·취소 주문이 섞인 원본을 분석 가능한 테이블로 정제 | Pandas, 데이터 품질 규칙, 재현 가능한 ETL | 541,909건 중 완전 중복 5,268건·고객 ID 결측 135,080건을 식별하고 목적별 3개 테이블 생성 | [README](01_data-cleaning/README.md) |
| 02. EDA | 시간·요일·날씨에 따른 공공자전거 수요를 운영 관점에서 해석 | Pandas, Seaborn, 시계열·교차 분석 | 18시 수요 피크, 평일 18시·주말 17시 피크, 5mm 이상 강수 시 수요 약 90.4% 감소 확인 | [README](02_eda-seoul-bike/README.md) |
| 03. SQL Analytics | 주문·고객·상품 데이터를 관계형 모델로 적재하고 비즈니스 지표를 SQL로 분석 | MySQL 8, ERD, CTE, Window Function, Pandas 검증 | 99,441명 고객·112,650개 주문상품 변환, 분석 SQL 10개와 월별 매출 대조 파이프라인 구성 | [README](03_sql-analysis/README.md) |
| 04. Customer Segmentation | 제한된 마케팅 예산에서 고객군별 프로모션 전략 수립 | RFM, K-Means, StandardScaler, Silhouette, ARI | 93,358명 분석, K=2 선택, 80% 표본·seed 10개 평균 ARI 0.991 | [README](04_customer-segmentation/README.md) |
| 05. Featured | 강남구 카페업의 매출·점포·유동인구를 비교해 후보 상권 탐색 | 공공데이터 결합, 지표 설계, Streamlit, Plotly | 2024~2025년 172,911건 통합, 유동인구는 높고 점포당 매출이 낮은 후보 상권 13곳 식별 | [README](05_commercial-district-dashboard/README.md) |

## Featured Project

### 강남구 카페업 상권 분석 대시보드

예비 자영업자와 상권 분석 담당자가 총매출만으로 판단하지 않고, 점포 수·점포당 추정매출·유동인구·경쟁도를 함께 비교하도록 설계했습니다.

- 2024 Q1 대비 2025 Q4: 추정매출 4.6% 증가, 점포 수 1.8% 감소, 점포당 추정매출 6.5% 증가
- 2025 Q4 강남구 커피·음료 76개 상권 중 유동인구는 중앙값 이상이지만 점포당 추정매출은 중앙값 미만인 후보 상권 13곳 식별
- 2025 Q3→Q4에 점포 수는 늘었지만 추정매출은 증가하지 않은 상권 26곳 확인

[프로젝트 상세 보기](05_commercial-district-dashboard/README.md)

## Engineering Focus

- **재현성:** 원본 데이터는 Git에서 제외하고, 다운로드·정제·분석 스크립트로 결과를 다시 만들 수 있게 구성
- **분석 정합성:** 취소 주문, 고객 ID 결측, 공간 단위 변경, 데이터 기간 차이처럼 결과 해석을 바꿀 수 있는 조건을 문서화
- **의사결정:** 그래프 자체보다 어떤 데이터를 근거로 어떤 다음 행동을 제안할 수 있는가에 초점
- **Backend 연결:** Olist 프로젝트는 MySQL 스키마, 적재 스크립트, SQL 분석과 Pandas 교차 검증으로 데이터 모델링 역량을 함께 다룸

## Learning Notes

[Clustering/Cluster_ShoppingMall.ipynb](Clustering/Cluster_ShoppingMall.ipynb)는 make_blobs를 사용한 K-Means 기초 실습입니다. 실제 거래 데이터 기반 프로젝트와 구분해 학습 기록으로 보존합니다.
