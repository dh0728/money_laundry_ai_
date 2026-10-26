# money_laundry_ai_
ibm aml 데이터 분석및 모델 실험 레포

# 데이터·산출물 위치

데이터(`datasets/`)와 산출물(`parquets/`)은 **GPU 서버에만 있다** (gitignore). 로컬 Windows 는 코드 편집용이라 이 폴더들이 없다.
코드는 `datasets/` 또는 `parquets/` 가 있는 폴더를 위로 찾아 루트로 잡으므로(`sampling_lib.find_root`) 아래 상대 경로 그대로 쓰면 된다.

| 위치 | 경로 |
|---|---|
| GPU 서버 (KubeSphere, Jupyter 루트 `/workspace`) | `/workspace/money_laundry_ai_/` |
| 서비스 저장소 (GPU, 나란히) | `/workspace/money_laundry/` |
| 로컬 (코드만) | `C:\Users\aica_\Desktop\money_laundry_ai_\` |
| 원본 CSV 로컬 사본 | `C:\Users\aica_\Desktop\ibm\datasets\archive\` |

```
/workspace/money_laundry_ai_/
├─ datasets/                          원본 (IBM AMLworld)
│   └─ <DATASET>_Trans.csv / <DATASET>_accounts.csv / <DATASET>_Patterns.txt     DATASET = HI-Small, HI-Large ...
├─ assets/binary_v10/                 v10 보정표 사본 (git 포함) currency_stats.parquet, amount_stats.parquet
└─ parquets/<DATASET>/                예: parquets/HI-Large/
    ├─ transactions.parquet           01 출력. 중복 제거·정렬 거래 (edge_id, ts, src, dst, paid, ..., is_laundering, day)
    ├─ duplicates.parquet             01 출력. 제거된 완전 중복 행
    ├─ kyc.parquet                    01 출력. 노드(계좌) 표 node_id, bank_id, account, entity_*, country
    ├─ patterns.parquet               01 출력. Patterns.txt 파싱 + edge_id 매칭 (attempt_id, pattern_type)
    ├─ summary.json                   01 요약 수치
    ├─ features/                      02 출력 (v10 과 같은 값)
    │   ├─ edge_features.parquet      엣지 피처 17개 + 원시값
    │   ├─ node_features.parquet      노드 피처 19개(+3)
    │   ├─ stats.parquet, amount_stats.parquet, features.json
    ├─ features_<tag>/                02 를 다른 인자로 돌린 실험판
    ├─ features_train|val|test/       05 실험용 구간별 피처 (02 를 TAG=train/val/test, CONTEXT_END=2022-10-02/10-16/10-30)
    │                                  노드 피처가 그 구간 끝까지의 거래로만 계산됨 (미래 누수 방지). 기본 features/ 는 전체 기간이라 05 에선 안 씀
    ├─ analysis/                      03·04 분석 결과 json/png
    └─ sampling/                      05 다운샘플링 실험
        ├─ cache/features_train|val|test/  GraphStore memmap 캐시 (구간 끝까지의 거래만, 각 8~10GB, adj_*.npy, node_x.npy, ready.json) — 피처가 바뀔 때만 삭제
        ├─ select/<이름>/             선택 결과 selected.parquet, meta.json, audit.json  (이름 예: random_r30, strata_r10_day-amount-cross_bank)
        ├─ val/<기간>_neg<rate>.parquet  고정 검증 표본 (모든 조건 공유) — 지우지 말 것
        ├─ runs/<조건이름>/           학습 결과 config.json, run.json, history.csv, val_scores.parquet, model.pt
        ├─ results/                   collect 출력 runs.csv, summary.csv, audit.csv, ratio_curves.png
        └─ duckdb_tmp/                DuckDB 스필 임시 폴더
```

- 05 시간 분할(`sampling_lib.TRAIN/VAL/TEST`): 학습 2022-08-08~10-02 (8주) · 검증 10-03~10-16 (2주) · 테스트 10-17~10-30 (2주). 11월 이후 미사용
- HI-Large 규모: 거래 179,702,079 (중복 150 제거), 노드 2,116,168, 기간 2022-08-01 ~ 2023-01-12, 라벨1 225,546 (패턴 매칭 137,936)
- 컨테이너 메모리 한도(cgroup) 약 27GB — `free` 의 503GB 는 호스트 값. DuckDB `memory_limit` 은 12GB 이하로
- 완료 표시 파일이 있으면 건너뛴다: select → `selected.parquet`, audit → `audit.json`, train → `run.json`. 다시 돌리려면 그 폴더 삭제


# 해결 해야 할것

클래스 불균형을 해결하기 위한 데이터 샘플링 데이터 전처리
지금 만들어진 피처가 점검

GATHER-SCATTER, SCATTER-GATHER을 fan-in, fan-out으로 분해를 했을때 우리가 의도한 대로 분해가 된다면
두개의 패턴을 없애고 fan-in, fan-out으로 분류한 다음에 에피소드로 묶을 수 있을 거 같은데 그게 가능 할지 확인

지금 피처를 누적을 하는게 매번 누적을 하다보니 시간도 오래걸리고 뒤로 갈수록 용량이 커져서 파일을 전송하는데 어려움이 생기는데 이걸 어떻게 해결 할 것인가
1. 백엔드 서버에서 입력 파일을 생성(피처 계산)해서 파일 저장 후 보내기
2. 추론서버에 파일을 저장해서 거기다가 갱신을 시키기
    - 동기화를 시키는게 중요할 듯(은행이 다시 보냈다던가 하는)

지금 데이터

# 궁금중

우리는 일반화성능을 높여야 하는데 이게 모델링을 하면 할수록 생성기 규칙을 찾아버리는 느낌이 드는데 이게 괜찮은 건지?

이진모델 피처랑 패턴 모델 피처랑 같을 필요가 있는지?

지금 