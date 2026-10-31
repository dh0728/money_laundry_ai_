# 유형(다중클래스) 분류 실험

이진 모델이 걸러낸 **의심 거래만 모은 그래프**를 입력으로, 각 거래(엣지)의 유형을 9클래스로 맞힌다.
첫 실험(상한선)은 실제 세탁 거래만으로 그래프와 라벨을 만든다. 설계는 `docs/mult_classification/` 참고.

- 클래스 9개 = 패턴 8개(`FAN-OUT, FAN-IN, GATHER-SCATTER, SCATTER-GATHER, CYCLE, RANDOM, BIPARTITE, STACK`) + `UNMATCHED`(무패턴 세탁)
- 한 거래가 여러 시도에 들어 있으면 번호가 가장 작은 시도의 유형 (`n_attempts`, `types` 로 표시)
- 분할 기본값은 이진 실험과 같다: 학습 08-08~10-02 / 검증 10-03~10-16 / 테스트 10-17~10-30, 11월 미사용

| 파일 | 내용 |
|---|---|
| `mc_lib.py` | 공통: 경로, 클래스 표, 분할(`make_windows`, `assign_period`), 분석 함수, 그림 |
| `01_prepare_laundering.ipynb` | `transactions.parquet` + `patterns.parquet` → `parquets/<DATASET>/multiclass/` (laundering, attempts, daily_all, summary) |
| `02_split_pattern_analysis.ipynb` | 구간별 클래스 수, 경계 잘림, `DATA_END` 끝 잘림, 시간 추이, 분할 후보 비교 → `results/<DATASET>/02_split_analysis/` |
| `03_laundering_graph_check.ipynb` | 0단계 점검: 세탁만 그래프의 이웃 종류(쉬운 규칙 여부), 연결 요소, 계좌 공유 → `results/<DATASET>/03_graph_check/` |

## 실행 순서 (GPU 서버 Jupyter)
1. `01_prepare_parquet.ipynb` 가 `<DATASET>` 으로 돌아 있어야 함 (`transactions.parquet`, `patterns.parquet`)
2. `01_prepare_laundering.ipynb` — 전체 거래 2회 스캔(세탁 필터, 일별 집계), 나머지는 세탁 거래만
3. `02_split_pattern_analysis.ipynb` — 설정 셀의 `TRAIN/VAL/TEST`, `DATA_END` 만 바꿔 다시 돌린다
4. `03_laundering_graph_check.ipynb` — 01 출력만 읽음 (scipy 필요)
