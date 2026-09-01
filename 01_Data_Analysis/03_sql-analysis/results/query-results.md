# SQL 분석 실행 결과

MySQL 8.4 컨테이너에 Olist 변환 데이터를 적재한 뒤 실행한 결과다.
각 표는 재현성을 위한 상위 5행 미리보기이며, `row_count`는 전체 결과 행 수다.

## 01. Monthly revenue

- 전체 결과 행 수: 23
- 미리보기: 상위 5행

| revenue_month | revenue | orders |
| --- | --- | --- |
| 2016-09 | 143.46 | 1 |
| 2016-10 | 46490.66 | 265 |
| 2016-12 | 19.62 | 1 |
| 2017-01 | 127482.37 | 750 |
| 2017-02 | 271239.32 | 1653 |

## 02. Top 10 products by revenue

- 전체 결과 행 수: 10
- 미리보기: 상위 5행

| product_id | product_category | revenue | item_count |
| --- | --- | --- | --- |
| bb50f2e236e5eea0100680137654686c | health_beauty | 67258.03 | 194 |
| d1c427060a0f73f6b889a5c7c61f2ac4 | computers_accessories | 58957.31 | 332 |
| 6cdd53843498f92890544667809f1595 | health_beauty | 57933.73 | 153 |
| 99a4788cb24856965c36a24e339b6058 | bed_bath_table | 49907.50 | 477 |
| 3dd2a17168ec895c781a9191c1e95ad7 | computers_accessories | 47876.06 | 272 |

## 03. Top 10 customers by lifetime spend

- 전체 결과 행 수: 10
- 미리보기: 상위 5행

| customer_unique_id | delivered_orders | lifetime_revenue |
| --- | --- | --- |
| 0a0a92112bd4c708ca5fde585afaa872 | 1 | 13664.08 |
| da122df9eeddfedc1dc1f5349a1a690c | 2 | 7571.63 |
| 763c8b1c9c68a0229c42c9fc6f662b93 | 1 | 7274.88 |
| dc4802a71eae9be1dd28f5d788ceb526 | 1 | 6929.31 |
| 459bef486812aa25204be022145caa62 | 1 | 6922.21 |

## 04. Repeat customer rate

- 전체 결과 행 수: 1
- 미리보기: 상위 5행

| customers_with_orders | repeat_customers | repeat_customer_rate_pct |
| --- | --- | --- |
| 93358 | 2801 | 3.00 |

## 05. Monthly revenue from new versus existing customers

- 전체 결과 행 수: 43
- 미리보기: 상위 5행

| revenue_month | customer_type | revenue |
| --- | --- | --- |
| 2016-09 | new | 143.46 |
| 2016-10 | new | 46490.66 |
| 2016-12 | new | 19.62 |
| 2017-01 | existing | 19.62 |
| 2017-01 | new | 127462.75 |

## 06. Customers without orders. Olist source can return zero rows.

- 전체 결과 행 수: 0
- 미리보기: 상위 5행

| customer_unique_id | customer_city | customer_state |
| --- | --- | --- |

## 07. Average order value by product category

- 전체 결과 행 수: 72
- 미리보기: 상위 5행

| product_category | category_orders | avg_order_value |
| --- | --- | --- |
| computers | 177 | 1290.11 |
| small_appliances_home_oven_and_coffee | 72 | 683.97 |
| home_appliances_2 | 227 | 520.74 |
| agro_industry_and_commerce | 177 | 430.53 |
| musical_instruments | 611 | 330.91 |

## 08. Dormant customers: no purchase in the 90 days before the latest data date

- 전체 결과 행 수: 74,899
- 미리보기: 상위 5행

| customer_unique_id | last_purchase_at | days_since_last_purchase |
| --- | --- | --- |
| 830d5b7aaa3b6f1e9ad63703bec97d23 | 2016-09-15 12:16:38 | 713 |
| 87776adb449c551e74c13fc34f036105 | 2016-10-03 22:31:31 | 695 |
| 8d3a54507421dbd2ce0a1d58046826e0 | 2016-10-03 22:06:03 | 695 |
| 61db744d2f835035a5625b59350c6b63 | 2016-10-03 21:13:36 | 695 |
| 7390ed59fa1febbfda31a80b4318c8cb | 2016-10-03 22:44:10 | 695 |

## 09. Monthly product revenue rank

- 전체 결과 행 수: 65
- 미리보기: 상위 5행

| revenue_month | revenue_rank | product_id | product_category | revenue |
| --- | --- | --- | --- | --- |
| 2016-09 | 1 | 5a6b04657a4c5ee34285d1e4619a96b4 | health_beauty | 143.46 |
| 2016-10 | 1 | eba7488e1c67729f045ab43fac426f2e | perfumery | 2423.71 |
| 2016-10 | 2 | 4fee671ea459ebc96546523917e254a5 | consoles_games | 1962.32 |
| 2016-10 | 3 | fd8a5b9a8a79d7ba0739d69be5dc5aa1 | watches_gifts | 1423.55 |
| 2016-12 | 1 | f5d8f4fbc70ca2a0038b9a0010ed5cb0 | fashion_bags_accessories | 19.62 |

## 10. Customer RFM

- 전체 결과 행 수: 93,358
- 미리보기: 상위 5행

| customer_unique_id | recency_days | frequency | monetary |
| --- | --- | --- | --- |
| 0a0a92112bd4c708ca5fde585afaa872 | 334 | 1 | 13664.08 |
| da122df9eeddfedc1dc1f5349a1a690c | 515 | 2 | 7571.63 |
| 763c8b1c9c68a0229c42c9fc6f662b93 | 45 | 1 | 7274.88 |
| dc4802a71eae9be1dd28f5d788ceb526 | 563 | 1 | 6929.31 |
| 459bef486812aa25204be022145caa62 | 35 | 1 | 6922.21 |
