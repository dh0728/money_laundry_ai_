# money_laundry_ai_ — IBM AML 데이터 분석·모델 실험 저장소

AML 모니터링 서비스(`money_laundry` 저장소)에 들어가는 GNN 모델을 **데이터 쪽에서 검증·실험**하는 곳이다.
서비스 코드는 여기 없다. 여기서 만든 결론(피처 창, 집계 기간, 보정표 등)이 서비스의 `backend/inference-server/models/binary_v10`
또는 `backend/api-server/worker/features/materials.py` 변경 근거가 된다.

## 환경

- 폴더 구성은 로컬과 GPU가 같다 — 두 저장소가 나란히 있다:
  - 로컬(Windows) `C:\Users\aica_\Desktop\money_laundry_ai_`, `C:\Users\aica_\Desktop\money_laundry`
  - GPU 서버(KubeSphere, Jupyter in Docker) `/workspace/money_laundry_ai_`, `/workspace/money_laundry` (Jupyter 루트 = `/workspace`)
  - 데이터(`datasets/`)와 산출물(`parquets/`)은 GPU에만 있다. 로컬은 코드 편집용.
- 서비스 저장소 `money_laundry` (GitHub dh0728/money_laundry, dev 브랜치) — 대조할 원본:
  `backend/inference-server/models/binary_v10/preprocess.py` (피처·그래프), `materials.py` (재료 정규화),
  `backend/api-server/worker/features/materials.py` (백엔드가 보내는 재료). 상대 경로로는 `../money_laundry/...`.
- 원본 데이터 로컬 사본: `C:\Users\aica_\Desktop\ibm\datasets\archive\` (초기 EDA 코드 `ibm\code\`).
- 이 저장소 GitHub: dh0728/money_laundry_ai_ (master).
- 파이썬 패키지: `duckdb`(≥1.1), `pyarrow`, `pandas`. 추론서버 가상환경과 **별개**(그쪽은 torch·torch-geometric).
- 경로는 코드가 `datasets/` 폴더를 위로 찾아 `ROOT` 를 잡으므로 어디서 열어도 된다.

## 폴더

```
datasets/            원본 CSV (gitignore). HI/LI × Small/Medium/Large × Trans.csv / accounts.csv / Patterns.txt
parquets/<DATASET>/  01 출력 (gitignore): transactions.parquet, duplicates.parquet, kyc.parquet, patterns.parquet, summary.json
parquets/<DATASET>/features[_<tag>]/   02 출력: edge_features.parquet, node_features.parquet, stats.parquet, amount_stats.parquet, features.json
assets/binary_v10/   v10 보정표 사본 (currency_stats.parquet, amount_stats.parquet) — money_laundry 추론서버 assets 와 동일 바이트
code/
  01_prepare_parquet.ipynb   CSV → 중복 제거·정렬 parquet + kyc + patterns
  02_build_features.ipynb    v10 피처 (엣지 17 · 노드 19) 계산
  03_same_minute_analysis.ipynb  같은 분(EXCLUDE='1 minute') 사각 분석 — 라벨별 "같은 분 2건 이상" 비율·lift, 패턴별, 02 피처 사각, gap 분포
                             → parquets/<DATASET>/analysis/same_minute*.json + png
  03_1_same_minute_analysis_noReinvestment.ipynb  03 과 동일, Reinvestment 제외판 (결과 _noReinvestment 접미사; 03 json 있으면 끝에 비교표)
  04_node_count_drift.ipynb  카운트형 노드 피처 8개의 월별 드리프트 — 누적 창 vs 고정 창(w90), 세탁/정상 계좌 단변량 AUC·중앙값 추이, 엣지 수준 AUC
                             → parquets/<DATASET>/analysis/node_drift*.json + png
  05_sampling_select.py      다운샘플링 실험 ① 학습에 넣을 정상 target 선택 (none/random/hard/cssmc × ratio) → parquets/<DATASET>/sampling/select/
  05_sampling_audit.py       ①b 선택 결과 통계 감사(학습 무관): 피처 KL/PSI/KS, 일별 보존율, 시간대 JSD, 세탁계좌 연루 비율, 양성 1홉 보존 → select/<name>/audit.json
  05_sampling_train.py       ② 실험 조건 1개 학습·평가 = select × apply(loss|struct) × reweight(ipw|none) × pi_eff × seed → sampling/runs/<run>/run.json
  05_sampling_collect.py     ③ runs/*/run.json → sampling/results/ (runs.csv, summary.csv, ratio_curves.png)
  sampling_lib.py            05 공통: GraphStore(v10 과 같은 2홉 그래프, memmap 캐시), BinaryGNN(v10 network.py 와 같은 층), 가중 지표
                             설계 메모: 지도 샘플링(loss)=그래프 유지·loss target 만 선택, 구조 샘플링(struct)=adjacency 에서도 제거(구조-A).
                             r(정상:양성)과 pi_eff(양성 손실 비중)를 분리 — pos_weight 는 pi_eff 에서 역산. 검증은 양성 전부 + 정상 2% 가중 표본(공유)
  0N_*.ipynb                 그 외 실험 (01·02 출력을 읽는다)
```

## 참고 문서

- **`AML 데이터 파이프라인 추적.html`** (저장소 루트) — 서비스 `money_laundry` 의 데이터 흐름을 코드 발췌와 함께 정리한 페이지.
  은행 CSV 업로드(S3) → 수신 워커 검수·AES-GCM 암호화(`private.bank_reports`) → 통합·가명화(`ledger.transactions`, `core.owners/accounts`)
  → 입력 동결(`analysis.input_transactions/input_ledger`) → FEATURES(재료 3파일) → INFERENCE(S3 게시·PUT) → GPU 추론서버(binary_v10)
  → SCORES 까지, 단계별 실제 코드·DB 전체 컬럼·"한 행 따라가기" 추적기(fixture 거래를 각 단계 저장 형태로 재현)·GNN v10 전처리/피처/2홉 그래프/네트워크.
  서비스 쪽 테이블 컬럼, 어떤 코드가 어느 단계에서 도는지, 재료 파일 포맷, 피처 SQL 출처가 궁금하면 이 파일을 먼저 본다.
  브라우저로 열면 되고(코드 하이라이팅만 CDN이라 오프라인이면 색이 빠짐), 기준은 money_laundry 커밋 00fcc79(2026-10-19).
- `code/*.ipynb` 의 마크다운 셀 — 각 단계의 규칙과 원본 코드 대응을 적어 두었다.

## 데이터 규칙 (서비스와 맞춘 것)

- 원본 헤더는 `Account` 열이 두 번 → 위치로 읽고 이름을 직접 준다 (`ELEVEN`).
- 은행 코드 `"020"` → `20` 정수. 계좌 노드 키 = `(bank_id, account)`. `node_id` 는 그 정렬 순서 0..N-1.
- 완전 중복(원본 11열 동일)은 **대표 1건**만 쓴다 (`duplicate_group_id IS NULL OR edge_id = duplicate_group_id`).
  서비스 통합 단계는 반복 행을 발생 건수대로 보존하지만, 모델 전처리(v10)는 대표만 쓰므로 여기선 모델 규칙.
- `edge_id` 는 `(ts, 11열)` 정렬 순서 — 파일 순서와 무관하게 재현된다.
- `ts` 는 원본 벽시계(분 단위, 초 없음, 타임존 없음). 서비스는 이걸 Asia/Seoul 로 읽는다.
- `entity_type` 은 `"Corporation #123"` 꼴 이름에서 추출, `country` 는 `"Canada Bank #27"` 에서 추출. 없으면 NULL → 피처에선 `'UNKNOWN'`.
- `Patterns.txt` 의 BEGIN 줄은 **콜론이 없는 타입이 있다** (BIPARTITE, SCATTER-GATHER, STACK). 파서 정규식은 콜론을 선택으로.
- 패턴 이름: 파일은 `GATHER-SCATTER`/`SCATTER-GATHER`, v10 `feature_schema.json` 은 `G-SCATTER`/`S-GATHER`. 매핑 필요.

## 피처 규칙

- `02` 의 SQL 은 `money_laundry/backend/inference-server/models/binary_v10/preprocess.py` 와 **글자 단위로 같다**.
  HI-Small 2022-09-01~03 에서 원본 `prepare_snapshot` 과 36개 피처 전부 `max|diff| = 0` 확인(2026-10-20).
  기본 인자(`EXCLUDE='1 minute'`, `HOUR_WINDOW='60 minutes'`, `STATS='package'`, 하한 없음)면 추론서버와 같은 값.
- 엣지 시간 창은 `RANGE … AND INTERVAL '1 minute' PRECEDING` → 같은 분 거래는 어떤 엣지 시간 피처에도 안 들어간다.
  노드 피처는 `context_end` 까지 **전체 누적**(대상 거래 시점 무관). 집계 하한 날짜는 서비스·모델 어디에도 없다.
- 보정표 `stats(scale, med, iqr)`: `scale = median(paid)`, `med = ln 2`, `iqr = IQR(ln(1+paid/scale))` 로 역추적됨.
  `amount_stats(m, sd)` 는 `ln(1+paid)` 의 평균·표준편차. 둘 다 학습 데이터로 고정된 값.
- 피처 SQL 을 바꾸는 실험은 `02` 의 설정 셀 인자(`EXCLUDE`, `WINDOW_DAYS`, `CONTEXT_END`, `STATS`, `TAG`)로 하고,
  결과는 `features_<tag>/` 로 분리한다. 기본 출력 `features/` 는 원본과 같은 값만 둔다.

## 작업 방식

- 큰 데이터(Large: 1.8억 행, 4.8GB parquet)는 **DuckDB 안에서** 처리한다. 전체 테이블을 `pandas` 로 올리지 않는다
  (`pd.read_parquet(transactions.parquet)` 금지 — `LIMIT` 또는 집계 결과만 `.df()`).
- DuckDB 는 파일 DB(`work*.duckdb`)로 열어 스필을 허용하고, `memory_limit` 은 컨테이너 한도의 60~70%.
  `temp_directory` 는 디스크 여유 있는 곳. Large 전체 실행은 노트북보다 `.py` + `nohup` 이 안전.
- 중간 테이블은 VIEW 로 반복 조인하지 말고 TABLE 로 한 번 만들거나 parquet 를 다시 읽는다.
- 노트북은 "셀 단위 리뷰용", `.py` 는 "반복 실행용". 둘의 SQL 이 어긋나지 않게 한쪽을 고치면 다른 쪽도 맞춘다.
- 커밋·push 는 사용자가 한다. `parquets/`, `datasets/`, `*.duckdb`, `.ipynb_checkpoints/` 는 커밋하지 않는다.
- 설명은 한국어로 간단히. 명령은 GPU 컨테이너 터미널/노트북 어디서 실행하는지 명시.

## 현재 상태 (2026-10-20)

- HI-Large `parquets/HI-Large/` 완료: 거래 179,702,079(중복 150 제거), 노드 2,116,168, 기간 2022-08-01~2023-01-12,
  라벨1 225,546 중 패턴 파일 매칭 137,936 (8개 패턴 16,467 시도) → 나머지 87,610 은 unmatched 세탁.
- 다음: `02` 를 HI-Large 에 (먼저 `CONTEXT_END` 2주치로 시간·메모리 측정) → `03` 실험.
- 열린 질문: 집계 기간 하한(학습 창과 맞춰야 함, 미확인), 같은 분 사각이 라벨1에서 얼마나 큰가,
  `paid_currency_z` vs `amt_z_pay_ccy` 중복 여부, unmatched 세탁의 성격.
