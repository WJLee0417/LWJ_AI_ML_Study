# 데이터 계약

원본 문의 데이터는 `raw/inquiries.csv`에 UTF-8 CSV로 둔다. 기본 분할의 필수 열은 아래 두 개다.

```csv
text,label
배송이 아직 안 왔어요,delivery
환불은 언제 처리되나요,refund
```

`text`에는 전화번호·주소·주문번호·이메일 같은 직접 식별정보를 넣지 않는다. `label`은 안정적인 업무 분류 코드로 지정하고, 사용자 화면용 한국어 명칭은 향후 서빙 단계에서 별도 매핑한다.

`prepare_data.py`는 학습 전 `text`에서 이메일, 국내 전화번호, `주문번호`가 붙은 주문 식별자를 각각 `[EMAIL]`, `[PHONE]`, `[ORDER_ID]`로 치환한다. 이는 방어적 전처리일 뿐 원본 접근 통제를 대체하지 않는다. 라벨 코드와 경계는 [라벨 정의서](../docs/label-guide.md)를 따른다.

## 날짜 기반 평가용 열

새로운 표현에 대한 성능을 확인하려면 ISO-8601 형식의 `timestamp` 열을 추가한다.

```csv
text,label,timestamp
배송이 아직 안 왔어요,delivery,2026-01-15T09:30:00+09:00
환불은 언제 처리되나요,refund,2026-02-03T14:20:00+09:00
```

날짜 기반 분할은 과거 → validation → 미래 test 순서를 유지하며, 미래에만 처음 등장하는 라벨은 학습할 수 없으므로 실행을 중단한다.

`processed/`는 `prepare_data.py`가 생성한 고정 train/validation/test 분할이며 Git에서 제외한다.

## AI Hub 소상공인 주문 문의 변환

AI Hub **소상공인 고객 주문 질의-응답 텍스트**의 압축을 `raw/Training/`, `raw/Validation/`에 푼 뒤 다음 명령을 실행한다.

```bash
python src/prepare_aihub_order_qa.py
```

변환기는 아래 규칙을 적용한다.

- `발화자=c` 및 `QA여부=q`만 선택해 점원 답변을 제외한다.
- `발화문 → text`, 매핑된 `인텐트 → label`, `상담번호 → group_id`로 변환한다.
- `배송_* → delivery`, `교환|반품|환불_* → refund`, `제품_* → product`만 첫 실험 라벨로 사용한다. `account`에 해당하는 일관된 원본 인텐트는 없어 제외한다.
- Training은 `상담번호` 단위 group-stratified train/validation 분할, 원본 Validation은 최종 test holdout으로 쓴다.
- 이메일·전화번호·주문번호는 `[EMAIL]`, `[PHONE]`, `[ORDER_ID]` 토큰으로 치환한다.

분할 수량·그룹 중복 검사·원본 날짜 충족률·PII 치환 건수는 `processed/split-summary.json`에서 확인한다.
