# 이커머스 주문·고객·상품 SQL 분석

## 문제 정의

쇼핑몰 주문 데이터를 MySQL 관계형 모델로 적재한 뒤, 매출·고객·상품 지표를 SQL로 분석한다. Python은 원본 CSV 정제와 SQL 결과 검증에 사용하고, 비즈니스 질문에 대한 답은 SQL로 작성한다.

## 데이터

- 출처: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- 제공 범위: 2016~2018년 브라질 이커머스 주문 데이터
- 사용 원본: customers, orders, order_items, products, product-category translation
- 분석 매출: 배송 완료(delivered) 주문의 price + freight_value

이 데이터는 상품 카테고리와 고객 고유 ID를 제공하므로 고객 재구매·카테고리·객단가 분석에 적합하다. 원본 및 변환 CSV는 Git에 커밋하지 않는다.

## 데이터 모델

~~~mermaid
erDiagram
    CUSTOMERS ||--o{ ORDERS : places
    ORDERS ||--|{ ORDER_ITEMS : contains
    PRODUCTS ||--o{ ORDER_ITEMS : appears_in
    CUSTOMERS {
        varchar customer_id PK
        varchar customer_unique_id
        varchar customer_city
        char customer_state
    }
    PRODUCTS {
        varchar product_id PK
        varchar product_category
    }
    ORDERS {
        varchar order_id PK
        varchar customer_id FK
        varchar order_status
        datetime purchase_at
    }
    ORDER_ITEMS {
        varchar order_id PK,FK
        int order_item_id PK
        varchar product_id FK
        decimal price
        decimal freight_value
    }
~~~

독립 Mermaid 원본은 [docs/erd.mmd](docs/erd.mmd)에 있다. GitHub README에서 바로 렌더링되며, 필요하면 Mermaid Live Editor 등에서 PNG로 내보낼 수 있다.

![Olist SQL 분석 ERD](assets/erd.svg)

## 실행 순서

### 1. 원본 다운로드 및 변환

~~~powershell
python src/download_data.py
python src/transform_data.py
~~~

### 2. MySQL 8 컨테이너 준비

Docker Compose가 MySQL 8.4 컨테이너와 스키마 초기화를 제공한다. 비밀번호는 Git에 저장하지 않고 현재 PowerShell 세션에만 둔다.

~~~powershell
$env:MYSQL_PASSWORD = "local_app_password"
$env:MYSQL_ROOT_PASSWORD = "local_root_password"
$env:MYSQL_PORT = "3309"
docker compose up -d
docker compose ps
~~~

컨테이너가 healthy가 된 뒤, 같은 세션에 애플리케이션 접속 정보를 설정한다.

~~~powershell
$env:MYSQL_HOST = "127.0.0.1"
$env:MYSQL_PORT = "3309"
$env:MYSQL_DATABASE = "shopping_analytics"
$env:MYSQL_USER = "app_user"
~~~

### 3. 적재 → SQL 실행 → Pandas 검증

아래 순서는 포트폴리오 결과를 처음부터 재현한다. run_sql_analysis.py는 10개 쿼리를 실제 실행하고, 각 결과의 전체 행 수와 상위 5행을 results/query-results.md에 기록한다.

~~~powershell
python src/load_to_mysql.py --reset
python src/run_sql_analysis.py
python src/verify_monthly_revenue.py
~~~

월별 매출 검증이 끝나면 results/generated/monthly-revenue-verification.md에 PASS와 월별 대조표가 생성된다. 작업을 마친 뒤 컨테이너를 종료하려면 docker compose down을 실행한다.

## 분석 SQL 10개

| 번호 | 질문 | SQL 기능 |
| ---: | --- | --- |
| 01 | 월별 매출은 어떻게 변하는가? | DATE_FORMAT, 집계 |
| 02 | 매출 상위 상품은 무엇인가? | GROUP BY, LIMIT |
| 03 | 누적 구매액 상위 고객은 누구인가? | 조인, 집계 |
| 04 | 재구매 고객 비율은 얼마인가? | CTE, HAVING |
| 05 | 월별 신규·기존 고객 매출은? | Window Function |
| 06 | 주문 이력이 없는 고객은? | LEFT JOIN |
| 07 | 카테고리별 객단가는? | 주문 단위 재집계 |
| 08 | 최근 구매 기준 휴면 고객은? | 기준일 CTE, 날짜 계산 |
| 09 | 월별 상품 매출 순위는? | DENSE_RANK() |
| 10 | 고객별 RFM 기초 지표는? | 다중 CTE |

## 실행 검증 결과

MySQL 8.4 컨테이너에 344,483행을 적재하고 10개 SQL을 실행해 확인한 결과다.

- 배송 완료 주문 고객 93,358명 중 재구매 고객은 2,801명으로, 재구매율은 **3.00%**였다.
- 데이터 기준일 90일 이전의 마지막 구매 고객은 **74,899명**으로 집계됐다.
- 매출 상위 상품은 health_beauty 카테고리의 bb50f2e...로, 배송 완료 주문 기준 매출은 **67,258.03**이었다.
- Pandas와 MySQL의 23개월 월별 매출은 모든 월에서 차이 **0.00**으로 일치해 PASS했다.

전체 결과와 월별 대조 근거는 아래 검증 산출물에서 확인할 수 있다.

## 데이터 해석 시 주의점

- 주문 이력이 없는 고객은 원본이 주문 중심으로 수집됐기 때문에 결과가 0명일 수 있다. 쿼리는 CRM/회원 테이블로 확장 가능한 형태로 유지한다.
- customer_id는 주문 시점 고객 레코드이고, 재구매 분석은 customer_unique_id 기준으로 한다.
- 데이터 마지막 구매일을 분석 기준일로 사용하므로 휴면 고객은 데이터셋 내부 상대 기준이다.
- 카테고리 번역이 없는 상품은 unknown으로 보존한다.

## 검증 산출물

- [10개 SQL 실행 결과](results/query-results.md): 쿼리별 전체 결과 행 수와 상위 5행
- [월별 매출 대조 결과](assets/monthly-revenue-verification.md): Pandas–MySQL 오차 0.01 이하 PASS 증빙
- [Docker Compose 정의](compose.yaml): MySQL 8.4, healthcheck, 스키마 초기화
