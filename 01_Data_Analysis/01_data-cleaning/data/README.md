# 데이터 관리

## 디렉터리

- `raw/`: 출처에서 내려받은 변경하지 않은 원본 파일
- `processed/`: `src/preprocess.py`가 생성한 정제 데이터

두 디렉터리의 실제 데이터 파일은 `.gitignore`로 제외한다. 원본 데이터의 출처·다운로드 방법·변환 규칙은 프로젝트 루트 `README.md`에 기록한다.

## 재현 절차

```powershell
python src/download_data.py
python src/preprocess.py
```

원본 파일의 SHA-256 해시는 다운로드 스크립트 실행 결과로 확인할 수 있다.
