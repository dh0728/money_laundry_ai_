# 이진 분류 실험 — 선행 연구 방식 재현 후 피처를 하나씩 추가

05 샘플링 실험(v10 구조, target 당 2홉 서브그래프, 36피처)과 별개로, 선행 연구(Multi-GNN / FraudGT / GFP)가 쓴 방식 그대로 시작해서
피처를 단계적으로 추가하며 성능 변화를 본다. 분할·검증 표본·지표는 05 와 같아서 결과를 나란히 비교할 수 있다.

| 노트북 | 피처 | 비고 |
|---|---|---|
| `01_basic_features_baseline.ipynb` | 기본 4개 (Timestamp, Amount Received, Received Currency, Payment Format). GNN 은 + port numbering 2개 | LightGBM / XGBoost (기본 4) / GINe / GINe+EU (6) |
| 02 (예정) | + v10 엣지 피처 17개 | |
| 03 (예정) | + v10 노드 피처 19개 | |
| 04 (예정) | + 시간 창 fan, 계좌 금액 통계 (GFP 식) | |
| 05 (예정) | + 사이클·scatter-gather 참여 횟수 | |

## 공통 설정
- 분할: 학습 2022-08-08~10-02 / 검증 ~10-16 / 테스트 ~10-30 (`sampling_lib.TRAIN/VAL/TEST`)
- 그래프: 구간 끝까지의 거래 전부 (검증·테스트 그래프는 이전 거래 포함, Multi-GNN 규칙). `to` / `rev_to` hetero (역방향 MP), ego ID
- GNN 학습: `LinkNeighborLoader` 로 target 2048개가 서브그래프 공유, 이웃 2홉 × 방향별 100, 가중 CE [1, 6.27], 에폭 = 배치 256개 × 400 (약 2바퀴), 검증 4 에폭마다
- 불균형: 가중 손실 / scale_pos_weight 만. 다운샘플링 없음
- 검증 표본: `parquets/<DATASET>/sampling/val/*.parquet` (양성 전부 + 정상 2%, 05 와 공유). 테스트는 구간 전체
- 메모리 (한도 약 25GB): 그래프를 도착 노드순 정렬 + `is_sorted=True`, float16/uint8, 그래프는 한 번에 하나만 (`use_graph`). 노트북 4절 설명 참고
- 결과: `results/<DATASET>/<실험명>/` (results.json, compare.csv, *_history.csv, curves.png)

## 처음 돌릴 때
1. `DATASET = 'HI-Small'`, `SMOKE = True` 로 끝까지 한 번 (01_prepare_parquet 을 HI-Small 로 먼저 실행해 둘 것)
2. 문제 없으면 `HI-Large`, `SMOKE = False`. 메모리가 부족하면 `NUM_NEIGHBORS`, `BATCH_SIZE` 부터 줄인다
