# 03 노트북(메모리 파이프라인 검증됨)을 바탕으로 04_amount_stats.ipynb 를 만든다
import json, sys, copy

src_path, dst_path = sys.argv[1], sys.argv[2]
nb = json.load(open(src_path, encoding='utf-8'))
cells = nb['cells']
MD = lambda s: dict(cell_type='markdown', metadata={}, source=s)
CODE = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)
S = lambda i: ''.join(cells[i]['source'])

def find(prefix):
    for i in range(len(cells)):
        if S(i).startswith(prefix):
            return i
    raise KeyError(prefix)

new = {}
new[0] = MD("""# 이진 분류 실험 ④ — 계좌 금액 분포 피처 추가 (amount stats)
1단계 피처 사다리의 세 번째 칸. 03(기본 4 + port 2 + pair 4 + 활동 8) 위에 **이 거래 금액이 그 계좌의 최근 금액 분포에서 어디에 있나** 피처 8개를 더한다. 모델은 GINe 고정, 설정 동일.

| 피처 | 뜻 (전부 그 거래 시각 기준 과거 24시간, 같은 분 제외) |
|---|---|
| `src_amt_mean_1d` / `src_amt_std_1d` / `src_amt_max_1d` | 송신 계좌가 최근 24시간에 보낸 금액의 평균 / 표준편차 / 최대 (ln(1+x)) |
| `src_amt_ratio_1d` | 이 거래 금액이 그 평균의 몇 배인가: ln(1+금액) − ln(1+평균). 이력 없으면 0 |
| `dst_amt_mean_1d` / `dst_amt_std_1d` / `dst_amt_max_1d` / `dst_amt_ratio_1d` | 수신 계좌가 최근 24시간에 받은 금액 쪽 같은 4개 |

03 의 `*_amt_1d` 는 **합계**(활동량)였고, 여기는 **분포**(평소 금액 대비 이상치). 금액 자체(`amount_received`)는 01 부터 있었다.

**가설**: 03 에서 유일하게 나빠진 **UNMATCHED**(패턴 없는 세탁, 양성의 41%) 를 건드리는지가 관전 포인트. 패턴 피처가 아니라 "이 계좌답지 않은 금액" 피처라서, 패턴 recall 은 03 수준을 유지하고 UNMATCHED 의 argmax recall(0.013) · 평균 점수(0.063) 가 오르면 성공. 금액을 쪼개 보내는 STACK · 큰 돈이 한 번에 들어오는 GATHER-SCATTER 도 볼 것.

**메모리** (03 파이프라인 그대로): ① 피처 parquet 은 tx 올리기 전에 ② 피처 float16 으로 붙임 ③ z-norm 행렬은 디스크 memmap ④ 트리 입력은 `rows_matrix`, 테스트 예측은 `predict_rows`.
03 보다 피처 8열이 늘어 트리 입력 X_train 이 6.5 → 9.8GB (float32, 1억 × 24). 03 의 트리 단계 RSS 15.4 → 약 19GB 예상, 학습 단계는 memmap 이라 03 과 같음(피크 22GB). 트리에서 넘치면 X_train 을 만들기 전에 `del test_df` 류로 줄이지 말고, 0 절의 `TREE_FEATURES` 에서 03 의 1h 두 개를 빼서 다시.

**누수 규칙**: 거래 시각 기준 과거만, 같은 분 제외, z-norm 은 학습 구간 통계. 공통 코드 `bc_lib.py`""")

i = find('# ---- 실험 설정')
new[i] = CODE("""# ---- 실험 설정 ------------------------------------------------------------
RUN_NAME = 'amount_stats'         # results/<DATASET>/amount_stats/
PREV_RUN = 'account_activity'     # 비교 기준 (03)
cfg = B.Config(dataset='HI-Large', smoke=False)
B.seed_all(cfg.seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = cfg.out_dir(RUN_NAME)

# 앞 단계 피처 (02·03 parquet 재사용) + 이번 단계 피처
PAIR_FEATURES = ['pair_prior_count', 'pair_gap_minutes', 'pair_count_1h', 'reverse_history_count']
PAIR_PATH = cfg.feat_dir / 'pair_history.parquet'
ACT_FEATURES = ['src_out_1h', 'src_out_1d', 'src_uniq_dst_1d', 'src_out_amt_1d', 'dst_in_1h', 'dst_in_1d', 'dst_uniq_src_1d', 'dst_in_amt_1d']
ACT_PATH = cfg.feat_dir / 'account_activity.parquet'
STAGE_FEATURES = ['src_amt_mean_1d', 'src_amt_std_1d', 'src_amt_max_1d', 'src_amt_ratio_1d',
                  'dst_amt_mean_1d', 'dst_amt_std_1d', 'dst_amt_max_1d', 'dst_amt_ratio_1d']
STAGE_PATH = cfg.feat_dir / 'amount_stats.parquet'
PREV_FEATURES = PAIR_FEATURES + ACT_FEATURES
GNN_EDGE_FEATURES = B.BASIC_EDGE_FEATURES + B.PORT_FEATURES + PREV_FEATURES + STAGE_FEATURES   # FEAT16 열 순서 (26). port 는 4,5 번
TREE_FEATURES = B.BASIC_EDGE_FEATURES + PREV_FEATURES + STAGE_FEATURES                         # 트리는 port 없이 (24)
for p_ in (PAIR_PATH, ACT_PATH):
    assert p_.exists(), f'앞 단계 parquet 이 없다: {p_}'
DEVICE, OUT_DIR, len(GNN_EDGE_FEATURES), len(TREE_FEATURES)""")

i = find('## 1. 이번 단계 피처 계산')
new[i] = MD("## 1. 이번 단계 피처 계산 — 계좌 금액 분포 (tx 를 올리기 **전에**)\n03 과 같은 이유: DuckDB 창 함수(12GB) 와 tx 를 동시에 올리면 한도를 넘는다. 한 번 만들면 이 절은 바로 지나간다")
new[i + 1] = MD("""- 03 의 창과 같은 꼴(`RANGE … 24 hours PRECEDING AND 1 minute PRECEDING`) 에 집계만 `avg / stddev_pop / max` 로 바꾼 것. src 파티션 한 번, dst 파티션 한 번
- `ratio` 는 저장 단계에서 `ln(1 + received) − ln(1 + avg)` 로 계산. 이력이 없으면(avg NULL) 0 — 03 의 `*_out_1d = 0` 이 "이력 없음" 을 이미 말해 주므로 여기서는 중립값
- 금액은 `received` (Amount Received). 결과는 `features_bc/amount_stats.parquet` (edge_id 순)""")
new[i + 3] = CODE("""# 이전 실행이 COPY 도중 죽었으면 불완전한 parquet 이 남아 있을 수 있다 → 행 수가 tx 와 다르면 지우고 다시 만든다
if STAGE_PATH.exists():
    n_stage = con.execute(f"SELECT count(*) FROM read_parquet({L.q(STAGE_PATH)})").fetchone()[0]
    n_tx = con.execute(f"SELECT count(*) FROM read_parquet({L.q(cfg.tx_path)})").fetchone()[0]
    if n_stage != n_tx:
        print(f'불완전한 {STAGE_PATH.name} ({n_stage:,} / {n_tx:,} 행) → 삭제 후 재생성'); STAGE_PATH.unlink()
if not STAGE_PATH.exists():
    t0 = time.time()
    con.execute(f\"\"\"
        CREATE OR REPLACE TABLE tx_min AS
        SELECT edge_id, src, dst, ts, received FROM read_parquet({L.q(cfg.tx_path)})\"\"\")
    # (a) 송신 계좌 기준 — 최근 24시간에 보낸 금액의 분포
    con.execute(\"\"\"
        CREATE OR REPLACE TABLE src_win AS
        SELECT edge_id, received,
               avg(received)        OVER w1d AS amt_mean,
               stddev_pop(received) OVER w1d AS amt_std,
               max(received)        OVER w1d AS amt_max
        FROM tx_min
        WINDOW w1d AS (PARTITION BY src ORDER BY ts RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 minute' PRECEDING)\"\"\")
    print(f'  src_win {time.time()-t0:.0f}s')
    # (b) 수신 계좌 기준 — 최근 24시간에 받은 금액의 분포
    con.execute(\"\"\"
        CREATE OR REPLACE TABLE dst_win AS
        SELECT edge_id,
               avg(received)        OVER w1d AS amt_mean,
               stddev_pop(received) OVER w1d AS amt_std,
               max(received)        OVER w1d AS amt_max
        FROM tx_min
        WINDOW w1d AS (PARTITION BY dst ORDER BY ts RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 minute' PRECEDING)\"\"\")
    print(f'  dst_win {time.time()-t0:.0f}s')
    # (c) ln 변환 + ratio 계산 후 저장 (edge_id 순). 이력 없음(NULL) → mean/std/max 0, ratio 0
    con.execute(f\"\"\"
        COPY (
            SELECT s.edge_id,
                   ln(1 + coalesce(s.amt_mean, 0))::FLOAT                                                        AS src_amt_mean_1d,
                   ln(1 + coalesce(s.amt_std, 0))::FLOAT                                                         AS src_amt_std_1d,
                   ln(1 + coalesce(s.amt_max, 0))::FLOAT                                                         AS src_amt_max_1d,
                   (CASE WHEN s.amt_mean IS NULL THEN 0 ELSE ln(1 + s.received) - ln(1 + s.amt_mean) END)::FLOAT AS src_amt_ratio_1d,
                   ln(1 + coalesce(d.amt_mean, 0))::FLOAT                                                        AS dst_amt_mean_1d,
                   ln(1 + coalesce(d.amt_std, 0))::FLOAT                                                         AS dst_amt_std_1d,
                   ln(1 + coalesce(d.amt_max, 0))::FLOAT                                                         AS dst_amt_max_1d,
                   (CASE WHEN d.amt_mean IS NULL THEN 0 ELSE ln(1 + s.received) - ln(1 + d.amt_mean) END)::FLOAT AS dst_amt_ratio_1d
            FROM src_win s JOIN dst_win d USING (edge_id)
            ORDER BY s.edge_id
        ) TO {L.q(STAGE_PATH)} (FORMAT PARQUET, COMPRESSION ZSTD)\"\"\")
    con.execute("DROP TABLE src_win; DROP TABLE dst_win; DROP TABLE tx_min")
    print(f'{STAGE_PATH.name} 생성 {time.time()-t0:.0f}s')
con.execute(f"SELECT count(*), {', '.join(f'avg({c}) AS avg_{c}' for c in STAGE_FEATURES)} FROM read_parquet({L.q(STAGE_PATH)})").df().T""")
new[i + 4] = CODE("""# 창 동작 확인 — 하루에 거래 8건쯤 보낸 계좌 하나의 송금 타임라인. mean/max 가 과거 24시간 금액으로만 움직이고, ratio 가 (이번 금액 vs 평균) 인지
# 주의: 1.8억 건끼리 조인하지 않는다 → 그 계좌의 edge_id 만 골라 피처 parquet 에서 읽는다
acct = con.execute(f"SELECT src FROM read_parquet({L.q(cfg.tx_path)}) GROUP BY src, day ORDER BY abs(count(*) - 8), src LIMIT 1").fetchone()[0]
ids = con.execute(f"SELECT edge_id, ts, dst, received FROM read_parquet({L.q(cfg.tx_path)}) WHERE src = {acct} ORDER BY ts LIMIT 40").df()
feat = con.execute(f"SELECT edge_id, src_amt_mean_1d, src_amt_max_1d, src_amt_ratio_1d FROM read_parquet({L.q(STAGE_PATH)}) WHERE edge_id IN ({', '.join(map(str, ids.edge_id))})").df()
tl = ids.merge(feat, on='edge_id').sort_values('ts')
tl['mean_1d (금액)'] = np.expm1(tl['src_amt_mean_1d']).round(2); tl['max_1d (금액)'] = np.expm1(tl['src_amt_max_1d']).round(2)   # ln(1+x) 를 되돌려 금액으로
tl['ratio (배)'] = np.exp(tl['src_amt_ratio_1d']).round(2)
tl""")

i = find('## 2. 데이터 준비')
new[i] = MD("## 2. 데이터 준비 — 거래 전체 + port + 앞 단계 피처(02·03) + 이번 피처\n전부 edge_id 순 parquet 이라 조인 없이 열만 붙인다 (`attach_parquet`, float16)")
i = find('con.close(); con = L.connect')
new[i] = CODE("""con.close(); con = L.connect(threads=8, memory='3GB')      # 창 함수가 잡아 둔 DuckDB 버퍼 반납. 이후 쿼리는 작아서 3GB 면 충분
con.register('currency_codes', currency_codes); con.register('format_codes', format_codes)   # 코드표는 연결마다 다시 등록
tx = B.load_tx_lowmem(con, cfg, port_path=PORTS_PATH)                      # 기본 9열 + port 2열 — parquet 에서 열 하나씩
B.trim('(tx 기본+port 적재 후)')
tx = B.attach_parquet(tx, PAIR_PATH, PAIR_FEATURES, dtype=np.float16)     # 02
tx = B.attach_parquet(tx, ACT_PATH, ACT_FEATURES, dtype=np.float16)       # 03
tx = B.attach_parquet(tx, STAGE_PATH, STAGE_FEATURES, dtype=np.float16)   # 이번 피처
B.trim('(피처 20열 붙인 후)'); tx.shape""")

i = find('in_train = B.window_mask')
new[i] = CODE(S(i).replace('(18열 6.5GB 가 RSS 에서 빠짐)', '(26열 9.3GB 가 RSS 에서 빠짐)'))
i = find('# 트리 모델')
new[i] = MD("# 트리 모델 (기본 4 + pair 4 + 활동 8 + 이번 피처 8)")
i = find('# 배치 하나 꺼내서')
new[i] = CODE(S(i).replace('엣지 피처 18열', '엣지 피처 26열'))
i = find('## 14. 다음')
new[i] = MD("""## 14. 다음
- 05: GFP 모티프 (사이클·scatter-gather 참여 수, snapml GraphFeaturePreprocessor) → CYCLE, SCATTER-GATHER
- v10 36피처 참조 런 → 1단계 정리 (피처 사다리 표 + UNMATCHED 추이)
- 서비스 프로토콜 테스트 (`test_gnn_daily`: 그날 끝까지 그래프 + 그날 target) 를 02·03·04 체크포인트로""")

out = []
for j, c in enumerate(cells):
    c = copy.deepcopy(new.get(j, c))
    if c['cell_type'] == 'code':
        c['outputs'] = []; c['execution_count'] = None
    out.append(c)
nb['cells'] = out
json.dump(nb, open(dst_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('cells', len(out), '| replaced', sorted(new))
