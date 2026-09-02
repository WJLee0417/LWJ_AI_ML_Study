# 데이터 계약

`raw/`에는 클래스별 원본 이미지를 넣고, `processed/`는 `src/prepare_data.py`가 생성한다. 두 디렉터리의 실제 데이터는 Git에서 제외한다.

이미지는 한 파일에 하나의 주된 폐기물만 포함해야 하며, 중복·거의 같은 연속 촬영본은 같은 split에 유지하는 것이 바람직하다. 그렇지 않으면 test 성능이 실제 일반화 성능보다 높게 보일 수 있다.

연속 촬영·동일 물체·같은 영상 프레임처럼 관련된 이미지는 [dataset-manifest.example.csv](raw/dataset-manifest.example.csv)를 복사해 `group_id`로 묶는다. 실제 manifest는 `data/raw`에 두고 아래 명령으로 분할한다.

```bash
python src/prepare_data.py --manifest data/raw/dataset-manifest.csv
```

그룹 분할은 같은 `group_id`의 이미지를 하나의 split에만 배치한다. 그룹은 하나의 라벨에만 속해야 하며, 클래스마다 기본 70/15/15 비율을 위해 최소 7개 고유 그룹이 필요하다.
