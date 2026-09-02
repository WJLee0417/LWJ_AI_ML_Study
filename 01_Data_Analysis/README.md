# Data Analysis

데이터 품질을 먼저 점검하고, 탐색적 분석과 SQL 검증을 거쳐 고객 세분화·상권 비교 같은 의사결정 근거로 연결하는 분석 프로젝트 모음이다. 그래프나 집계값을 결과의 끝으로 두지 않고, 어떤 데이터 범위에서 무엇을 알 수 있으며 다음 행동 전에 무엇을 더 검증해야 하는지를 함께 기록한다.

이 문서는 전체 구조와 현재 재현된 분석 결과를 요약한다. 세부 데이터 사전, 실행 순서, 쿼리, 시각화, 생성 보고서는 각 하위 프로젝트 README와 results·assets 폴더를 기준으로 확인한다. 원본 데이터와 생성된 대용량 CSV는 Git에 포함하지 않는다.

## 한눈에 보기

| 영역 | 분석 질문 | 핵심 방법 | 현재 확인 결과 |
| --- | --- | --- | --- |
| [01 Data Cleaning](01_data-cleaning/README.md) | 주문 원본을 거래·고객 분석에 안전하게 쓸 수 있는가 | 품질 규칙, 재현 가능한 ETL, 손실 기록 | 541,909행 중 중복 5,268행·고객 ID 결측 135,080행 식별 |
| [02 Seoul Bike EDA](02_eda-seoul-bike/README.md) | 시간·요일·날씨에 따라 자전거 수요가 어떻게 달라지는가 | 시간 파생, 교차 집계, 시각화 | 18시 수요 피크, 5mm 이상 강수 시 평균 수요 약 90.4% 감소 |
| [03 SQL Analytics](03_sql-analysis/README.md) | 관계형 주문 데이터에서 매출·고객·상품 지표를 어떻게 검증하는가 | MySQL, CTE, Window Function, Pandas 대조 | 10개 분석 쿼리, 23개월 매출 대조 오차 0.00 |
| [04 Customer Segmentation](04_customer-segmentation/README.md) | 제한된 예산에서 고객군별 행동 전략을 어떻게 나눌 수 있는가 | RFM, K-Means, Silhouette, ARI | 93,358명 분석, K=2, 10개 seed 평균 ARI 0.991 |
| [05 Commercial Dashboard](05_commercial-district-dashboard/README.md) | 강남구 카페업의 추가 검토 상권을 어떻게 좁힐 수 있는가 | 공공데이터 결합, 지표 설계, Streamlit, Plotly | 2025 Q4 후보 상권 13곳, 점포 증가·매출 미증가 상권 26곳 |

## 전체 구조

~~~text
01_Data_Analysis/
├── Clustering/                       # K-Means 합성 데이터 기초 노트북
├── 01_data-cleaning/                 # Online Retail 품질 점검·정제
├── 02_eda-seoul-bike/                # 서울 공공자전거 수요 EDA
├── 03_sql-analysis/                  # Olist 관계형 모델·SQL 분석
├── 04_customer-segmentation/         # Olist RFM 기반 세분화
├── 05_commercial-district-dashboard/ # 강남구 카페업 Streamlit 대시보드
└── README.md                          # 전체 목적, 결과, 분석 원칙
~~~

| 경로 | 역할 | Git 관리 원칙 |
| --- | --- | --- |
| src/ | 다운로드, 정제, 적재, 분석, 고객 특성 생성 코드 | 포함 |
| sql/ 또는 queries/ | 분석 질문에 대응하는 SQL | 포함 |
| app.py | Streamlit 대시보드 진입점 | 포함 |
| data/README.md, data dictionary | 원본 범위·컬럼·결합 기준 | 포함 |
| results/ | Markdown, CSV 등 재현 가능한 수치 근거 | 포함 |
| assets/ | 차트, ERD, 대시보드 화면, 보고서 템플릿 | 포함 |
| data/raw/, data/processed/ | 원본과 생성된 대용량 데이터 | 제외 |

## 분석 흐름과 공통 원칙

### 1. 분석 전 데이터 품질과 관측 단위를 확인한다

중복, 결측, 취소, 0 이하 수량·단가처럼 결과를 바꿀 수 있는 문제를 먼저 수량화한다. 주문 라인, 주문, 고객, 시간별 집계, 상권·분기처럼 각 데이터의 관측 단위가 다르므로, 합계·평균·재구매율을 계산하기 전에 어떤 단위에서 집계하는지 명시한다.

### 2. 제외하지 않은 정보와 제외한 정보를 구분한다

취소 주문은 원본 데이터에서 의미 있는 상태이므로 정제 테이블에는 상태로 보존한다. 다만 매출·고객 분석에는 유효 일반 주문만 사용한다. 고객 ID 결측 행도 거래 분석에서는 남기지만 고객 단위 분석에서는 제외하며, 이러한 기준과 행 수 변화는 보고서에 남긴다.

### 3. 상관관계와 의사결정을 구분한다

시간·날씨·유동인구·점포 수와 매출의 관계는 운영 가설을 세우는 근거이지 인과 효과의 증명은 아니다. 예를 들어 유동인구가 많은 상권이 곧 창업 추천 지역이라는 뜻은 아니며, 임대료·업종 구성·체류 시간·경쟁 구조를 추가로 확인해야 한다.

### 4. SQL 결과는 독립 도구로 대조한다

Olist 분석은 Python으로 원본을 정제하고 MySQL에서 집계·조인·윈도우 함수를 실행한다. 월별 매출처럼 핵심 집계는 Pandas 결과와 대조해 계산·적재·쿼리의 정합성을 확인한다. 단순히 쿼리가 실행되는 것과 수치가 검증된 것은 다르다.

### 5. 세분화는 행동 전략의 출발점이지 고객 등급이 아니다

RFM 군집은 고객 행동의 유사성을 요약하는 비지도 분석이다. 군집 번호는 고객의 본질적 가치나 미래 행동을 보장하지 않는다. 캠페인 효과는 실제 같은 고객·같은 시점의 데이터와 대조군을 둔 실험으로 확인해야 한다.

~~~text
원본 데이터
  → 스키마·품질·관측 단위 점검
  → 목적별 정제 테이블 생성
  → EDA와 SQL 집계·대조
  → 고객·상권 단위 특징과 비교 지표 생성
  → 후보군·운영 가설 도출
  → 외부 요인·기간 차이·실험으로 추가 검증
~~~

## 프로젝트별 내용과 결과

### 01. Data Cleaning — Online Retail 주문 데이터 정제

UCI Online Retail 원본을 거래 분석과 고객 분석에 사용할 수 있는 테이블로 나눈다. 분석 기간은 2010-12-01부터 2011-12-09까지이며, 관측 단위는 주문 전체가 아니라 주문 상품 라인이다. 원본 파일을 재다운로드한 뒤 같은 전처리 코드로 결과를 만들 수 있도록 구성했다.

| 품질 항목 | 판단·처리 기준 | 확인 결과 |
| --- | --- | ---: |
| 원본 행 수 | 전체 주문 상품 라인 | 541,909 |
| 완전 중복 | 모든 컬럼이 같은 행은 한 행만 유지 | 5,268행 (0.97%) |
| 취소 주문 | InvoiceNo의 C 접두사 또는 음수 수량 | 10,587행 (1.95%) |
| 고객 ID 결측 | 거래 분석에는 보존, 고객 분석에서는 제외 | 135,080행 (24.93%) |
| 유효 거래 | 일반 주문 중 Quantity·UnitPrice가 양수 | 524,878행 |

전처리는 cleaned 전체 테이블, 거래 분석용 테이블, 고객 분석용 테이블을 각각 생성한다. 취소 주문을 원본에서 삭제하지 않고 is_cancellation 상태로 남기는 이유는 원본 의미를 보존하면서도 매출·고객 분석의 범위를 분명히 하기 위해서다.

~~~powershell
cd 01_data-cleaning
pip install -r requirements.txt
python src/download_data.py
python src/preprocess.py
~~~

정제 단계별 행 수·결측·중복·취소·유효 거래 현황은 [데이터 품질 보고서](01_data-cleaning/assets/data-quality-report.md)와 generated report에서 확인한다. 취소와 반품의 업무 맥락, 고객 ID 결측으로 인한 고객 분석의 표본 편향, 이상치 처리 기준은 이 데이터만으로 확정하지 않는다.

### 02. Seoul Bike EDA — 서울 공공자전거 수요 패턴

UCI Seoul Bike Sharing Demand의 2017-12-01부터 2018-11-30까지 시간 단위 데이터를 사용한다. 운영하지 않은 날은 수요 비교에서 제외하고, 날짜·시간을 결합해 월·요일·주말 여부를 파생한 뒤 시간대·요일·계절·강수량별 평균 대여 수를 비교한다.

| 분석 질문 | 현재 확인 결과 | 운영 관점의 해석 |
| --- | --- | --- |
| 언제 수요가 높은가 | 전체 평균 수요 최고 시간은 18시, 최고 요일은 Friday | 피크 이전에 업무·상업 지역의 자전거·거치 공간 점검 |
| 평일·주말 패턴은 다른가 | 평일 피크 18시, 주말 피크 17시 | 출퇴근·여가 수요에 맞춰 점검 시간 조정 |
| 날씨·계절의 관계는 | Summer 평균 수요가 가장 높고, 5mm 이상 강수 시 0mm 대비 평균 수요 약 90.4% 낮음 | 강수 예보일은 평시 재배치 기준을 그대로 적용하지 않음 |
| 분석 범위 | 운영일 시간 관측치 8,465건 | 대여소 단위 수요 불균형은 설명하지 못함 |

~~~powershell
cd 02_eda-seoul-bike
pip install -r requirements.txt
python src/download_data.py
python src/analyze.py
~~~

시간·요일 heatmap, 평일·주말 패턴, 계절·강수량 그래프와 운영 제안은 [프로젝트 README](02_eda-seoul-bike/README.md)와 [분석 보고서](02_eda-seoul-bike/results/generated/analysis-report.md)에서 확인한다. 시간 단위 집계와 단일 연도 자료이므로 개별 대여소 재배치 위치나 미래 연도의 동일 패턴을 단정하지 않는다.

### 03. SQL Analytics — Olist 주문·고객·상품 분석

Brazilian E-Commerce Public Dataset by Olist의 customers, orders, order_items, products, category translation 데이터를 MySQL 관계형 모델로 적재하고 SQL로 비즈니스 질문에 답한다. 분석 매출은 배송 완료 주문의 상품가와 배송비 합계이며, 고객 재구매는 customer_unique_id를 기준으로 계산한다.

| 확인 항목 | 결과 |
| --- | ---: |
| MySQL 적재 행 수 | 344,483 |
| 분석 SQL | 10개 |
| 월별 매출 기간 | 23개월 |
| Pandas–MySQL 월별 매출 차이 | 0.00, PASS |
| 배송 완료 주문 고객 | 93,358명 |
| 재구매 고객 / 재구매율 | 2,801명 / 3.00% |
| 데이터 기준일 90일 이전 마지막 구매 고객 | 74,899명 |
| 매출 상위 상품 | health_beauty 카테고리, 매출 67,258.03 |

10개 SQL은 월별 매출, 상위 상품·고객, 재구매율, 신규·기존 고객 매출, 주문 이력 없는 고객, 카테고리별 객단가, 휴면 고객, 월별 상품 매출 순위, RFM 기초 지표를 다룬다. CTE, LEFT JOIN, DATE_FORMAT, DENSE_RANK 등 관계형 분석 기능을 질문별로 사용한다.

~~~powershell
cd 03_sql-analysis
python src/download_data.py
python src/transform_data.py

$env:MYSQL_PASSWORD = "local_app_password"
$env:MYSQL_ROOT_PASSWORD = "local_root_password"
docker compose up -d
python src/load_to_mysql.py --reset
python src/run_sql_analysis.py
python src/verify_monthly_revenue.py
~~~

[실행 결과](03_sql-analysis/results/query-results.md)에는 각 쿼리의 전체 행 수와 상위 5행이, [월별 매출 검증](03_sql-analysis/assets/monthly-revenue-verification.md)에는 Pandas 대조 결과가 남아 있다. 주문 중심 원본이므로 주문 이력 없는 고객이 0명인 결과는 데이터 수집 범위의 특성일 수 있으며, CRM 회원 테이블 전체를 의미하지 않는다.

### 04. Customer Segmentation — Olist RFM 기반 세분화

SQL 분석에서 변환한 배송 완료 주문 데이터를 고객 한 명당 한 행으로 바꿔 RFM·객단가·선호 카테고리 특징을 만든다. K-Means 입력에는 로그 변환한 구매 빈도·누적 매출, 최근성, 객단가와 StandardScaler를 사용한다. 선호 카테고리는 명목형이므로 거리 계산에는 넣지 않고 군집 해석에만 사용한다.

| 선택 기준 | 결과 |
| --- | ---: |
| 분석 고객 수 | 93,358명 |
| 후보 K | 2~6 |
| 선택 K | **2** |
| 선택 근거 | K=3 이상은 최소 군집 비율 약 2.9%로 5% 기준 미충족 |
| 안정성 | 80% 표본·seed 10개 반복 평균 ARI **0.991** |

| 군집 | 고객 수 | 비율 | 평균 누적 매출 | 평균 최근성 | 해석·권장 방향 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 52,182명 | 55.9% | 68.94 | 240.7일 | 재활성화 쿠폰·선호 카테고리 리마인드 |
| 1 | 41,176명 | 44.1% | 287.12 | 232.1일 | 혜택 유지·신상품 우선 안내 |

군집 1의 평균 누적 매출은 군집 0보다 약 4.2배 높다. 그러나 이는 관측 기간의 행동 요약이며, 어떤 혜택이 매출을 높인다는 인과 결론은 아니다. K=4처럼 더 세분화된 군집이 있더라도 규모가 너무 작아 독립 캠페인으로 실행하기 어렵다면 선택하지 않는 기준을 적용했다.

~~~powershell
cd 03_sql-analysis
python src/download_data.py
python src/transform_data.py

cd ../04_customer-segmentation
pip install -r requirements.txt
python src/build_features.py
python src/segment_customers.py
~~~

이탈 위험과 RFM 군집을 함께 쓰는 캠페인 매트릭스는 [운영 설계 예시](04_customer-segmentation/results/campaign-priority-matrix.md)다. Olist와 Telco 이탈 예측은 다른 고객 집합이므로, 실제 고객 단위로 결합한 결과가 아니다. 운영 적용 전에는 같은 고객 식별자·같은 예측 시점에서 결합하고 A/B 테스트로 효과를 검증해야 한다.

### 05. Commercial District Dashboard — 강남구 카페업 상권 비교

서울시 상권분석서비스의 추정매출·점포·길단위인구·영역 데이터를 상권 코드·업종 코드·기준 년분기로 결합한다. 기본 화면은 강남구 커피·음료 업종이며, 비교 기간은 공간 단위가 같은 2024년 이후 분기 자료로 제한한다. 2024년 이전 자료는 표준단위구역 변경으로 같은 기준의 시계열 비교에 사용하지 않는다.

| 지표 | 정의 | 해석 시 주의 |
| --- | --- | --- |
| 분기 추정매출 | 상권·업종의 분기 카드 매출 기반 추정치 | 개별 점포의 실제 매출·손익이 아님 |
| 점포당 추정매출 | 추정매출 / 전체 점포 수 | 점포 수 0인 경우 제외 |
| 점포 수 증감률 | 직전 분기 대비 점포 수 변화 | 개업·폐업 원인과 구분 필요 |
| 유동인구 대비 매출 | 추정매출 / 총 유동인구 | 상관 지표이지 인과 아님 |
| 경쟁도 | 같은 분기·업종의 점포 수 백분위 | 높을수록 상대적 경쟁 밀집 |

강남구 커피·음료의 현재 결과는 아래와 같다.

| 항목 | 현재 확인 결과 |
| --- | --- |
| 통합 데이터 범위 | 2024~2025년, 172,911건 |
| 2024 Q1 → 2025 Q4 추정매출 | 1,004.6억 원 → 1,050.6억 원, 4.6% 증가 |
| 같은 기간 점포 수 | 1,717개 → 1,686개, 1.8% 감소 |
| 같은 기간 점포당 추정매출 | 5,851만 원 → 6,232만 원, 6.5% 증가 |
| 2025 Q4 비교 대상 | 강남구 커피·음료 76개 상권 |
| 유동인구 중앙값 이상·점포당 추정매출 중앙값 미만 후보 | 13곳 |
| 2025 Q3 → Q4 점포 수 증가·추정매출 미증가 상권 | 26곳 |

2025 Q4 상권 단위 비교에서 점포 수와 추정매출의 Pearson 상관계수는 0.794, 유동인구와 추정매출은 0.665였다. 반면 유동인구와 점포당 추정매출의 상관계수는 0.070으로, 유동인구가 많다는 사실만으로 점포별 매출 효율이 높다고 볼 수 없었다.

후보 상권 13곳은 창업 추천 목록이 아니다. 유동인구가 실제 매출로 충분히 전환되지 않는 이유를 추가로 살펴볼 1차 검토 대상이다. 임대료, 프랜차이즈 비중, 카페 수, 체류형 시설, 평일·주말·시간대별 매출 구성을 함께 확인해야 한다.

~~~powershell
cd 05_commercial-district-dashboard
pip install -r requirements.txt
python src/prepare_dataset.py
streamlit run app.py
~~~

대시보드는 상권·업종·분기 필터, KPI 카드, 시간대 매출 비중, 좌표 기반 상권 산점도, 점포·매출 증감 사분면, 후보 상권 CSV 다운로드를 제공한다. 화면과 상세 해석은 [프로젝트 README](05_commercial-district-dashboard/README.md)에서 확인한다. 길단위인구 원본은 현재 2025년 분기만 포함하므로 유동인구 기반 비교도 2025년 자료에 한정한다.

## 재현성과 결과물 확인

| 확인하려는 내용 | 우선 파일 |
| --- | --- |
| 정제 규칙·행 수 변화 | 01_data-cleaning results와 data-quality-report |
| 수요 패턴·운영 가설 | 02_eda-seoul-bike assets와 analysis-report |
| SQL 쿼리 결과·행 수 | 03_sql-analysis results/query-results.md |
| Pandas와 SQL 대조 | monthly-revenue-verification 보고서 |
| K 선택·군집 안정성·프로필 | 04_customer-segmentation segmentation-report와 assets |
| 캠페인 적용 시 유의점 | campaign-priority-matrix.md |
| 상권 후보 기준·대시보드 화면 | 05_commercial-district-dashboard README와 assets |

생성 보고서는 데이터·실행 환경에 따라 새로 만들어질 수 있다. README의 수치와 결과 파일이 다르면, 먼저 원본 데이터 버전·필터 조건·정제 실행일·공간·시간 범위를 비교한다.

## 데이터 관리과 해석 범위

- 원본 데이터는 각 프로젝트의 data/raw에 두고 Git에 올리지 않는다.
- 정제·변환 결과는 재생성할 수 있도록 코드와 데이터 사전·결합 키·제외 규칙을 함께 관리한다.
- 집계 결과·검증표·작은 Markdown 보고서·시각화는 분석 근거로 Git에 포함한다.
- SQL 접속 비밀번호, API 키, 개인 인증키는 환경 변수 또는 개인 설정으로 전달하고 저장소에 넣지 않는다.
- 데이터의 지역·기간·공간 단위가 바뀌면 이전 지표와 단순 비교하지 않는다.
- 고객 세분화, 이탈 위험, 상권 후보는 의사결정 검토를 돕는 분석 결과이며, 개인·고객·사업자에 대한 자동 처분이나 수익 보장이 아니다.

## 다음 정비 우선순위

1. 모든 분석 프로젝트에 원본 수집일, 코드 실행일, 행 수, 필터 조건을 담은 작은 실행 manifest를 추가한다.
2. Seoul Bike 분석은 다른 연도·대여소 단위 데이터가 확보되면 시간대 수요 패턴과 재배치 가설을 재검증한다.
3. Olist 분석은 주문 중심 원본의 한계를 보완할 CRM·캠페인·리뷰·배송 데이터가 있으면 고객 행동과 결과를 확장해 본다.
4. RFM 세분화는 실제 같은 고객 데이터에서 이탈 확률·캠페인 반응과 연결하고, 무작위 대조군을 둔 실험으로 전략 효과를 측정한다.
5. 상권 대시보드는 임대료·폐업·프랜차이즈 비중·시간대별 지표를 추가하고, 공간 단위 변경과 유동인구 기간 차이를 명시한 상태로 갱신한다.
