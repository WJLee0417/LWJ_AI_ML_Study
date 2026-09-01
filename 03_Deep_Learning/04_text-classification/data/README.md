# 데이터 계약

원본 문의 데이터는 `raw/inquiries.csv`에 UTF-8 CSV로 둔다. 필수 열은 아래 두 개다.

```csv
text,label
배송이 아직 안 왔어요,delivery
환불은 언제 처리되나요,refund
```

`text`에는 전화번호·주소·주문번호·이메일 같은 직접 식별정보를 넣지 않는다. `label`은 안정적인 업무 분류 코드로 지정하고, 사용자 화면용 한국어 명칭은 향후 서빙 단계에서 별도 매핑한다.

`processed/`는 `prepare_data.py`가 생성한 고정 train/validation/test 분할이며 Git에서 제외한다.
