# 04~06 · v10 피처 재현 → LightGBM (팀원 SALT-GNN 입력)

팀원 SALT-GNN(FraudGT 계열, daily-close) 이 무패턴 자금세탁을 거의 못 찾는다.
**같은 입력**으로 LightGBM 을 돌려 원인이 피처(신호 없음)인지 모델·학습(임계값 0.5, F1@0.5 선택, 6에폭)인지 가른다.

팀원 노트북 `01_원본전처리` 의 피처 정의(`raw_common.py`, `preprocessing.py`)를 그대로 옮겼고,
입력은 우리 `code/01_prepare_parquet.ipynb` 출력이다.

## 실행 순서 (GPU 서버 Jupyter, `code/nonpaten_vs_normal` 에서)

| 노트북 | 하는 일 | 출력 `parquets/HI-Large/nonpaten_vs_normal/v10_lgbm/` |
|---|---|---|
| `04_v10_edge_features.ipynb` | 엣지 17 + 결제유형 + port | `edge_features.parquet` |
| `05_v10_node_features.ipynb` | 계좌 19, 날짜별 마감 누적 스냅샷 | `nodes/<날짜>.npy`, `first_days/` |
| `06_v10_lgbm_inputs.ipynb` | 구간별 학습 입력 (엣지 + src·dst 노드 + 라벨) | `lgbm/{train,val,test}.parquet` |
| 07 (다음) | LightGBM 학습·평가, 무패턴 전용 지표 | |

필요: `parquets/HI-Large/{transactions,kyc,patterns}.parquet` (01_prepare_parquet). 공통 코드는 `dv_lib.py`.

## 팀원 코드와 같은 것

- 이력은 `2022-08-01` 부터 누적 (모든 피처가 그 거래 이전 거래만 사용)
- 엣지: 같은 분 제외 창, 어제까지 쌍·역방향 이력, 8/1~8/7 로 맞춘 통화별 금액 z (v10 패키지 보정표 아님)
- 노드: 그날 마감(당일 나중 거래 포함)까지 누적한 19개, 활동 없는 계좌 0
- 라벨 `cls`: 0 정상, 1~8 패턴(BIPARTITE, CYCLE, FAN-IN, FAN-OUT, GATHER-SCATTER, RANDOM, SCATTER-GATHER, STACK), 9 무패턴

## 다를 수 있는 것

- **분할은 우리 binary_classification 기간**: Train 8/8~10/2 · Val 10/3~10/16(같은 2% 검증 표본) · Test 10/17~10/30, 10/31·11월 미사용.
  팀원 GNN 과 비교하려면 팀원 `test_epoch*/scores.npy` 에서 10/17~10/30 거래만 골라 지표를 다시 낸다
- edge_id·node_id 번호 체계가 우리 `01_prepare_parquet` 것 → port 는 같은 분에 처음 생긴 쌍끼리의 순서만 다를 수 있음
- 금액 DOUBLE (팀원은 DECIMAL(38,8) 경유) → 마지막 자리 반올림 수준
- 04 노트북이 구간별 거래 수를 binary_classification 실험 수와 비교한다. 다르면 멈춘다(`STRICT`)

## 검증 (합성 데이터)

원본 형식 합성 CSV 2.3만 건(중복·같은 분 반복·자기 이체·보정 후 등장 통화·crypto 은행 포함)을
팀원 노트북 코드와 이 폴더 코드에 각각 넣어 비교:
엣지 17개·결제유형·port 전부 일치(차 0), 노드 19개 매일 전부 일치(최대 2.4e-7, float32 합산 순서),
구간별 거래 수·양성 cls 일치 (팀원 분할 기준으로 확인). 실제 HI-Large 전체에서는 아직 안 돌려 봤다.
