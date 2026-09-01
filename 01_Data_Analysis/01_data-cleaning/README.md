# Online Retail 주문 데이터 정제 및 품질 분석

## 문제 정의

온라인 주문 원본 데이터에는 취소 주문, 고객 식별자 결측, 중복 행처럼 분석 결과를 왜곡할 수 있는 품질 문제가 섞여 있을 수 있다. 이 프로젝트는 원본 주문 데이터를 **거래 분석과 고객 분석에 사용할 수 있는 테이블**로 정제하고, 적용한 규칙과 데이터 손실을 재현 가능하게 기록한다.

## 데이터

- 출처: [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- 원본 파일: `Online Retail.xlsx`
- 기간: 2010-12-01 ~ 2011-12-09
- 단위: 주문 상품 라인(item line)
- 라이선스·사용 조건: 데이터 출처 페이지를 따른다.

원본 파일과 생성 결과물은 용량 및 재현성 관리를 위해 Git에 커밋하지 않는다. 아래 명령으로 내려받은 뒤 처리한다.

```powershell
python src/download_data.py
python src/preprocess.py
```

## 원본 스키마

| 컬럼 | 설명 | 예상 품질 확인 항목 |
| --- | --- | --- |
| `InvoiceNo` | 주문 번호 | `C` 접두사는 취소 주문인지 확인 |
| `StockCode` | 상품 코드 | 특수 코드·결측 여부 |
| `Description` | 상품명 | 결측·표기 불일치 |
| `Quantity` | 주문 수량 | 0 이하 값과 극단값 |
| `InvoiceDate` | 주문 일시 | 날짜 변환 가능 여부 |
| `UnitPrice` | 단가 | 0 이하 값과 극단값 |
| `CustomerID` | 고객 식별자 | 결측 비율 |
| `Country` | 주문 국가 | 범주 수·표기 불일치 |

## 품질 문제와 처리 규칙

| 문제 | 판단 기준 | 처리 |
| --- | --- | --- |
| 완전 중복 행 | 모든 컬럼이 동일 | 한 행만 유지 |
| 취소 주문 | `InvoiceNo`가 `C`로 시작하거나 `Quantity < 0` | 삭제하지 않고 `is_cancellation`으로 표시 |
| 유효하지 않은 판매 | 일반 주문인데 `Quantity <= 0` 또는 `UnitPrice <= 0` | 거래 분석 대상에서 제외 |
| 고객 ID 결측 | `CustomerID`가 결측 | 거래 분석 데이터에는 보존, 고객 분석 데이터에서는 제외 |
| 날짜 | `InvoiceDate` | datetime으로 변환하고 연·월·요일·시간 파생 |
| 매출 | 유효 일반 주문 | `line_revenue = Quantity * UnitPrice` 생성 |

중요: 취소 주문은 원본 데이터의 의미 있는 상태이므로 원본 정제 테이블에서 제거하지 않는다. 매출·고객 분석 대상 테이블에서만 제외하며, 그 수와 금액 영향은 보고서에 남긴다.

## 생성 파일

| 파일 | 용도 |
| --- | --- |
| `data/processed/orders_cleaned.csv` | 중복 제거·형 변환·상태 플래그·파생 변수 적용 전체 테이블 |
| `data/processed/orders_for_transaction_analysis.csv` | 유효 일반 주문만 포함한 거래 분석 테이블 |
| `data/processed/orders_for_customer_analysis.csv` | 유효 일반 주문 중 고객 ID가 있는 고객 분석 테이블 |
| `results/generated/data-quality-report.md` | 실행 시 자동 생성되는 데이터 품질 보고서 |

## 결과 스냅샷

| 항목 | 결과 |
| --- | ---: |
| 원본 행 수 | 541,909 |
| 완전 중복 행 | 5,268 (0.97%) |
| 취소 주문 행 | 10,587 (1.95%) |
| 고객 ID 결측 | 135,080 (24.93%) |
| 거래 분석용 유효 주문 | 524,878 |

[전체 데이터 품질 보고서](assets/data-quality-report.md)

## 실행 환경

Python 3.11 이상을 권장한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_data.py
python src/preprocess.py
```

## 완료 기준

- 원본 행 수와 정제 단계별 행 수가 보고서에 기록된다.
- 결측·중복·취소·유효하지 않은 판매의 수와 비율을 확인할 수 있다.
- 원본을 다시 내려받아도 같은 명령으로 결과를 재생성할 수 있다.
- 거래 분석과 고객 분석에서 제외한 규칙을 구분해 설명할 수 있다.

## 한계

- 취소 주문과 반품의 전체 업무 맥락은 데이터만으로 확정할 수 없다.
- `CustomerID` 결측 주문을 제거하면 고객 분석에서 표본 편향이 생길 수 있다.
- 이상치는 자동 삭제하지 않는다. 이후 분석 목적에 맞춰 별도 기준을 세운다.
