# 데이터 사전

| 원본 컬럼 | 단위 | 범위 또는 예시 | 분석에서의 역할 |
| --- | --- | --- | --- |
| `Date` | 날짜 | `01/12/2017` | 관측 날짜·월·요일 파생 |
| `Rented Bike Count` | 대 | 0 이상 정수 | 분석 대상 수요 |
| `Hour` | 시 | 0~23 | 시간대 수요 분석 |
| `Temperature(°C)` | °C | 실수 | 보조 날씨 변수 |
| `Humidity(%)` | % | 0~100 | 보조 날씨 변수 |
| `Wind speed (m/s)` | m/s | 0 이상 | 보조 날씨 변수 |
| `Visibility (10m)` | 10m | 0 이상 | 보조 날씨 변수 |
| `Dew point temperature(°C)` | °C | 실수 | 보조 날씨 변수 |
| `Solar Radiation (MJ/m2)` | MJ/m² | 0 이상 | 보조 날씨 변수 |
| `Rainfall(mm)` | mm | 0 이상 | 강수 구간 분석 |
| `Snowfall (cm)` | cm | 0 이상 | 보조 날씨 변수 |
| `Seasons` | 범주 | Winter/Spring/Summer/Autumn | 계절별 비교 |
| `Holiday` | 범주 | Holiday/No Holiday | 휴일 여부 파생 |
| `Functioning Day` | 범주 | Yes/No | 운영하지 않은 날 제외 기준 |

## 파생 컬럼

| 컬럼 | 생성 규칙 | 용도 |
| --- | --- | --- |
| `datetime` | `Date` + `Hour` | 시간 순서·파생 변수 생성 |
| `month` | `datetime.month` | 월별 수요 비교 |
| `weekday` | 월요일~일요일 | 요일별 수요 비교 |
| `is_weekend` | 토·일이면 `Weekend` | 평일/주말 패턴 비교 |
| `rainfall_band` | 0, 0~1, 1~5, 5mm 이상 | 강수 강도별 수요 비교 |
