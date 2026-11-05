# 03 노트북(검증된 파이프라인)을 바탕으로 06_account_history.ipynb 를 만든다
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
new[0] = MD("""# 이진 분류 실험 ⑥ — 계좌 이력 · 평소 대비 피처 추가 (account history)
1단계 피처 사다리의 네 번째 칸 (04 금액 분포는 효과가 없어 제외, 03 위에 쌓는다). 03 의 창 피처(1h/24h) 가 못 보는 **"이 계좌가 얼마나 오래·많이 썼나, 평소와 다른가"** 를 더한다. v10 채택 41종 중 정상 vs 무패턴 분리도가 높았던 묶음(`log_out_count` AUC 0.85, `day_ccy_rate_vs_prior` 0.83, `minute_sender_vs_prior_hour` 교란 시 무패턴 R −4.4pt) 을 우리 규칙(과거만) 으로 다시 정의한 것.

| 피처 | 뜻 | v10 대응 |
|---|---|---|
| `src_hist_out` / `dst_hist_in` | 전날까지 누적 송금 / 수취 건수 | `log_out_count` / `log_in_count` |
| `src_hist_uniq_out` / `dst_hist_uniq_in` | 전날까지 누적 서로 다른 상대 수 | `log_out_neighbors` / `log_in_neighbors` |
| `src_active_days` / `dst_active_days` | 전날까지 활동한 날짜 수 | `log_active_days` |
| `src_rate_vs_prior` / `dst_rate_vs_prior` | 최근 24h 건수 vs 과거 일평균 건수: ln(1+24h) − ln(1+누적/활동일). 이력 없으면 0 | `day_ccy_rate_vs_prior` (과거만 쓰는 판) |
| `src_minute_burst` / `dst_minute_burst` | 같은 분 집중도: ln(1+60·같은 분 건수) − ln(1+직전 60분 건수) | `minute_sender/receiver_vs_prior_hour` |
| `src_gap_minutes` / `dst_gap_minutes` | 계좌의 직전 거래까지 log(2+분), 없으면 0 | `sender/receiver_gap_minutes` |
| (선택) `src_amount_seen` / `dst_amount_seen` | 이 계좌가 전날까지 같은 금액을 처리한 적 있나 (0/1) | `prior_ccy_amount_seen` |

전부 ln. 누적·amount_seen 은 **전날까지**(일 단위 ASOF, reverse_history 와 같은 꼴), 창·간격은 **거래 시각 기준 과거, 같은 분 제외**. 예외는 `minute_burst` 하나 — 분 단위 데이터라 같은 분 안의 순서가 없어 같은 분 건수 전체를 쓴다(v10 과 동일, 일 마감 서비스에선 문제 없음).

**가설**: 무패턴 세탁(UNMATCHED, 양성의 41%) 은 이력이 짧은 계좌·평소보다 갑자기 활발한 계좌에서 나온다 → UNMATCHED 의 argmax recall(03: 0.013) 과 평균 점수(0.063), 1% 컷(0.431) 이 오르면 성공. 매칭 패턴은 03 수준 유지.
**amount_seen 은 꺼 두고 시작** (`USE_AMOUNT_SEEN = False`): v10 에서 가장 강한 무패턴 신호(ACH AUC 0.89) 였지만 AMLworld 정상 거래의 금액 반복 규칙에 기댄 것이라 SAML-D 에서 붕괴. parquet 엔 계산해 두고, 켠 조건은 두 번째 런으로.

**메모리**: 03 파이프라인 + 피처 행렬을 학습 전에 GPU 로 (`store.features_to_gpu`). 18 + 12 = 30열 → GPU 10.8GB, 학습 9.5GB → V100 32GB 안 (다른 프로세스 11GB 가 남아 있으면 빠듯 — 학습 CUDA OOM 이면 `batch_size=1024, iter_per_epoch=512` 로). 호스트는 memmap 을 안 읽어 스래싱 없음. 추론 배치 2048.

**누수 규칙**: 거래 시각 기준 과거만(같은 분 제외; minute_burst 만 같은 분 포함), z-norm 은 학습 구간 통계. 공통 코드 `bc_lib.py`""")

i = find('# ---- 실험 설정')
new[i] = CODE("""# ---- 실험 설정 ------------------------------------------------------------
RUN_NAME = 'account_history'      # results/<DATASET>/account_history/
PREV_RUN = 'account_activity'     # 비교 기준 (03). 04 는 제외
cfg = B.Config(dataset='HI-Large', smoke=False, eval_batch_size=2048)
B.seed_all(cfg.seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = cfg.out_dir(RUN_NAME)
USE_AMOUNT_SEEN = False           # True 면 amount_seen 2열 포함 (두 번째 런)

# 앞 단계 피처 (02·03 parquet 재사용) + 이번 단계 피처
PAIR_FEATURES = ['pair_prior_count', 'pair_gap_minutes', 'pair_count_1h', 'reverse_history_count']
PAIR_PATH = cfg.feat_dir / 'pair_history.parquet'
ACT_FEATURES = ['src_out_1h', 'src_out_1d', 'src_uniq_dst_1d', 'src_out_amt_1d', 'dst_in_1h', 'dst_in_1d', 'dst_uniq_src_1d', 'dst_in_amt_1d']
ACT_PATH = cfg.feat_dir / 'account_activity.parquet'
STAGE_FEATURES = ['src_hist_out', 'src_hist_uniq_out', 'src_active_days', 'src_rate_vs_prior', 'src_minute_burst', 'src_gap_minutes',
                  'dst_hist_in', 'dst_hist_uniq_in', 'dst_active_days', 'dst_rate_vs_prior', 'dst_minute_burst', 'dst_gap_minutes']
OPT_FEATURES = ['src_amount_seen', 'dst_amount_seen']
if USE_AMOUNT_SEEN:
    STAGE_FEATURES = STAGE_FEATURES + OPT_FEATURES; RUN_NAME = RUN_NAME + '_amtseen'; OUT_DIR = cfg.out_dir(RUN_NAME)
STAGE_PATH = cfg.feat_dir / 'account_history.parquet'
PREV_FEATURES = PAIR_FEATURES + ACT_FEATURES
GNN_EDGE_FEATURES = B.BASIC_EDGE_FEATURES + B.PORT_FEATURES + PREV_FEATURES + STAGE_FEATURES   # FEAT16 열 순서 (30). port 는 4,5 번
TREE_FEATURES = B.BASIC_EDGE_FEATURES + PREV_FEATURES + STAGE_FEATURES                         # 트리는 port 없이 (28)
for p_ in (PAIR_PATH, ACT_PATH):
    assert p_.exists(), f'앞 단계 parquet 이 없다: {p_}'
DEVICE, OUT_DIR, len(GNN_EDGE_FEATURES), len(TREE_FEATURES)""")

i = find('## 1. 이번 단계 피처 계산')
new[i] = MD("## 1. 이번 단계 피처 계산 — 계좌 이력 · 평소 대비 (tx 를 올리기 **전에**)\n03 과 같은 이유: DuckDB 창 함수(12GB) 와 tx 를 동시에 올리면 한도를 넘는다. 한 번 만들면 이 절은 바로 지나간다")
new[i + 1] = MD("""- (a) **일 단위 계좌 상태**: (계좌, 날짜) 별 건수 → 누적 건수·누적 상대 수(그 쌍의 첫 거래일에 1)·활동 일수를 날짜순 누적. 송신 역할·수신 역할 따로. `reverse_history` 와 같은 꼴로 **ASOF (전날까지)** 붙인다 — 행 수가 (계좌 × 활동일) 이라 1.8억 조인이 아님
- (b) **창 함수** (03 과 같은 꼴): 같은 분 건수(`PARTITION BY src, ts`), 직전 60분·24h 건수, 직전 거래 시각(`UNBOUNDED … 1 minute PRECEDING`), 같은 금액의 첫 거래일(`PARTITION BY src, received`). src 한 번, dst 한 번
- (c) ln 변환·조합 후 저장 (edge_id 순). `rate_vs_prior` = ln(1+24h 건수) − ln(1+누적/활동일), `minute_burst` = ln(1+60·같은 분) − ln(1+직전 60분), `amount_seen` = 첫 거래일 < 대상일
- 결과는 `features_bc/account_history.parquet` (14열, amount_seen 포함)""")
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
        SELECT edge_id, src, dst, ts, day, received FROM read_parquet({L.q(cfg.tx_path)})\"\"\")
    # (a) 일 단위 계좌 상태 — 송신 역할 / 수신 역할. 그 다음 (계좌, 날짜) 마다 전날까지의 상태를 ASOF 로
    for role, acct_col, other_col in [('out', 'src', 'dst'), ('in', 'dst', 'src')]:
        con.execute(f\"\"\"
            CREATE OR REPLACE TABLE {role}_state AS
            WITH d AS (SELECT {acct_col} AS acct, day, count(*) AS n FROM tx_min GROUP BY 1, 2),
                 p AS (SELECT {acct_col} AS acct, day, count(*) AS new_partners
                       FROM (SELECT {acct_col}, {other_col}, min(day) AS day FROM tx_min GROUP BY 1, 2) GROUP BY 1, 2)
            SELECT d.acct, d.day, sum(d.n) OVER w AS cum_cnt, sum(coalesce(p.new_partners, 0)) OVER w AS cum_uniq, count(*) OVER w AS active_days
            FROM d LEFT JOIN p USING (acct, day)
            WINDOW w AS (PARTITION BY d.acct ORDER BY d.day ROWS UNBOUNDED PRECEDING)\"\"\")
        con.execute(f\"\"\"
            CREATE OR REPLACE TABLE {acct_col}_hist AS
            SELECT t.{acct_col} AS acct, t.day, s.cum_cnt, s.cum_uniq, s.active_days
            FROM (SELECT DISTINCT {acct_col}, day FROM tx_min) t
            ASOF LEFT JOIN {role}_state s ON t.{acct_col} = s.acct AND t.day > s.day\"\"\")
        con.execute(f"DROP TABLE {role}_state")
        print(f'  {acct_col}_hist {time.time()-t0:.0f}s')
    # (b) 창 함수 — 같은 분 건수, 직전 60분·24h 건수, 직전 거래 시각, 같은 금액의 첫 거래일
    for acct_col in ['src', 'dst']:
        con.execute(f\"\"\"
            CREATE OR REPLACE TABLE {acct_col}_win AS
            SELECT edge_id,
                   count(*) OVER (PARTITION BY {acct_col}, ts) - 1          AS same_min,
                   count(*) OVER w1h                                        AS n_1h,
                   count(*) OVER w1d                                        AS n_1d,
                   max(ts)  OVER wall                                       AS last_ts,
                   min(day) OVER (PARTITION BY {acct_col}, received)        AS amt_first_day
            FROM tx_min
            WINDOW w1h  AS (PARTITION BY {acct_col} ORDER BY ts RANGE BETWEEN INTERVAL '60 minutes' PRECEDING AND INTERVAL '1 minute' PRECEDING),
                   w1d  AS (PARTITION BY {acct_col} ORDER BY ts RANGE BETWEEN INTERVAL '24 hours'  PRECEDING AND INTERVAL '1 minute' PRECEDING),
                   wall AS (PARTITION BY {acct_col} ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 minute' PRECEDING)\"\"\")
        print(f'  {acct_col}_win {time.time()-t0:.0f}s')
    # (c) ln 변환·조합 후 저장 (edge_id 순)
    def cols(a, hist_name, uniq_name):
        return f\"\"\"
                   ln(1 + coalesce({a}h.cum_cnt, 0))::FLOAT                                                             AS {a}_{hist_name},
                   ln(1 + coalesce({a}h.cum_uniq, 0))::FLOAT                                                            AS {a}_{uniq_name},
                   ln(1 + coalesce({a}h.active_days, 0))::FLOAT                                                         AS {a}_active_days,
                   (CASE WHEN {a}h.active_days IS NULL THEN 0 ELSE ln(1 + {a}w.n_1d) - ln(1 + {a}h.cum_cnt / {a}h.active_days) END)::FLOAT AS {a}_rate_vs_prior,
                   (ln(1 + 60 * {a}w.same_min) - ln(1 + {a}w.n_1h))::FLOAT                                                  AS {a}_minute_burst,
                   (CASE WHEN {a}w.last_ts IS NULL THEN 0 ELSE ln(2 + date_diff('minute', {a}w.last_ts, t.ts)) END)::FLOAT   AS {a}_gap_minutes,
                   (CASE WHEN {a}w.amt_first_day < t.day THEN 1 ELSE 0 END)::FLOAT                                         AS {a}_amount_seen\"\"\"
    con.execute(f\"\"\"
        COPY (
            SELECT t.edge_id, {cols('src', 'hist_out', 'hist_uniq_out')}, {cols('dst', 'hist_in', 'hist_uniq_in')}
            FROM tx_min t
            JOIN src_win srcw USING (edge_id) JOIN dst_win dstw USING (edge_id)
            JOIN src_hist srch ON t.src = srch.acct AND t.day = srch.day
            JOIN dst_hist dsth ON t.dst = dsth.acct AND t.day = dsth.day
            ORDER BY t.edge_id
        ) TO {L.q(STAGE_PATH)} (FORMAT PARQUET, COMPRESSION ZSTD)\"\"\")
    con.execute("DROP TABLE src_win; DROP TABLE dst_win; DROP TABLE src_hist; DROP TABLE dst_hist; DROP TABLE tx_min")
    print(f'{STAGE_PATH.name} 생성 {time.time()-t0:.0f}s')
con.execute(f"SELECT count(*), {', '.join(f'avg({c}) AS avg_{c}' for c in STAGE_FEATURES + OPT_FEATURES)} FROM read_parquet({L.q(STAGE_PATH)})").df().T""")
new[i + 4] = CODE("""# 창 동작 확인 — 하루에 거래 8건쯤 보낸 계좌 하나의 송금 타임라인. hist_out 은 날짜가 바뀔 때만 늘고(전날까지), rate 는 24h 건수 vs 평균, burst 는 같은 분에 몰릴 때만 양수인지
# 주의: 1.8억 건끼리 조인하지 않는다 → 그 계좌의 edge_id 만 골라 피처 parquet 에서 읽는다
acct = con.execute(f"SELECT src FROM read_parquet({L.q(cfg.tx_path)}) GROUP BY src, day ORDER BY abs(count(*) - 8), src LIMIT 1").fetchone()[0]
ids = con.execute(f"SELECT edge_id, ts, dst, received FROM read_parquet({L.q(cfg.tx_path)}) WHERE src = {acct} ORDER BY ts LIMIT 40").df()
feat = con.execute(f"SELECT edge_id, src_hist_out, src_hist_uniq_out, src_active_days, src_rate_vs_prior, src_minute_burst, src_gap_minutes, src_amount_seen FROM read_parquet({L.q(STAGE_PATH)}) WHERE edge_id IN ({', '.join(map(str, ids.edge_id))})").df()
tl = ids.merge(feat, on='edge_id').sort_values('ts')
for c in ['src_hist_out', 'src_hist_uniq_out', 'src_active_days']:
    tl[c.replace('src_', '') + ' (수)'] = np.expm1(tl[c]).round(0).astype(int)      # ln(1+x) 를 되돌려
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
tx = B.attach_parquet(tx, STAGE_PATH, STAGE_FEATURES, dtype=np.float16)   # 이번 피처 (amount_seen 은 USE_AMOUNT_SEEN 일 때만)
B.trim(f'(피처 {len(PREV_FEATURES) + len(STAGE_FEATURES)}열 붙인 후)'); tx.shape""")

i = find('in_train = B.window_mask')
new[i] = CODE(S(i).replace('(18열 6.5GB 가 RSS 에서 빠짐)', '(30열 10.8GB 가 RSS 에서 빠짐)'))
i = find('# 트리 모델')
new[i] = MD("# 트리 모델 (기본 4 + pair 4 + 활동 8 + 이번 피처 12)")
i = find('results = {}')
new[i] = CODE("""results = {}
tree_cols = [GNN_EDGE_FEATURES.index(c) for c in TREE_FEATURES]
train_mask = in_train; train_rows = np.flatnonzero(train_mask)
# X_train(1억 × 28 float32 = 11.5GB) 을 만들지 않는다 — LightGBM Sequence / XGBoost DataIter 로 memmap 을 묶음으로 읽어 bin 만 남긴다 (04 와 같음)
y_train = tx['y'].to_numpy()[train_mask]
X_val, y_val, w_val = B.rows_matrix(feat_z, val_idx, tree_cols), val_df['y'].to_numpy(), val_df['w'].to_numpy()
test_pred = lambda m: B.predict_rows(m, feat_z, test_idx, tree_cols)     # 테스트 행렬도 안 만들고 묶음 예측
B.trim('(트리 입력 준비 후)'); len(train_rows), X_val.shape""")
i = find('lgb_model, xgb_model = B.train_trees(')
new[i] = CODE(S(i).replace('B.train_trees(X_train, y_train, X_val, y_val, w_val, test_pred, val_df, test_df, cfg, results)',
                            'B.train_trees_lowmem(feat_z, train_rows, tree_cols, y_train, X_val, y_val, w_val, test_pred, val_df, test_df, cfg, results)'))
i = find('# 트리 테스트 점수도 저장')
new[i] = CODE(S(i).replace('del X_train, y_train, X_val; B.trim', 'del y_train, X_val; B.trim'))

i = find('# feat_z 가 이미 memmap 이라')
new[i] = CODE("""# 그래프 저장소 + 피처 행렬을 GPU 로. 04 테스트에서 memmap 스래싱(호스트 25GB 한도) 으로 멈춘 뒤, 학습부터 GPU 피처를 쓴다
# 30열 float16 = 10.8GB. 학습 GPU 피크 9.5GB 와 합쳐 V100 32GB 안. 학습 중 CUDA OOM 이면 cfg.batch_size=1024, iter_per_epoch=512 (에폭당 target 수 동일)
store = B.GraphStore(tx, feat_z, cfg, train_rows=train_rows, val_idx=val_idx, test_idx=test_idx,
                     port_cols=[GNN_EDGE_FEATURES.index('in_port'), GNN_EDGE_FEATURES.index('out_port')])
del tx, feat_z; B.trim('(tx, feat_z 삭제 후)')
store.features_to_gpu(DEVICE)
print(f'GPU 할당 {torch.cuda.memory_allocated() / 1e9:.1f} GB'); B.trim('(피처 GPU 이동 후)')
store.edge_dim""")
i = find('# 배치 하나 꺼내서')
new[i] = CODE(S(i).replace('엣지 피처 18열', '엣지 피처 30열'))

# 테스트 전 정리 셀 (04 에서 넣었던 것) 은 03 기반이라 없음 → 테스트 직전에 학습 전용 메모리 정리만 추가
i = find('gine_score = B.test_gnn(')
cleanup = CODE("""# 테스트 전 정리 — LightGBM 부스터가 학습 Dataset(bin) 을 붙들고 있고, 학습 행 인덱스도 테스트엔 필요 없다
for n in ['train_rows', 'train_mask', 'lgb_model', 'xgb_model', 'X_val', 'y_train']:
    globals().pop(n, None)
store.rows.pop('train', None)
gc.collect(); torch.cuda.empty_cache(); B.trim('(테스트 전 정리 후)')""")
new[i] = [cleanup, copy.deepcopy(cells[i])]

i = find('# 혼동행렬 — GINe')
new[i] = CODE(S(i).replace("title='GINe (pair_history)'", "title=f'GINe ({RUN_NAME})'"))

i = find('## 14. 다음')
new[i] = MD("""## 14. 다음
- amount_seen 조건 (`USE_AMOUNT_SEEN = True`) 한 번 더 — v10 최강 무패턴 신호의 효과와 일반화 위험을 분리해서 기록
- 05: GFP 모티프 (2홉 경로·사이클 수, snapml) → CYCLE, SCATTER-GATHER. 06 결과에 따라 03 또는 06 위에
- 1단계 정리 → 2단계 모델 비교 (HI-Small, timestamp 를 시간대로 교체한 설정)""")

out = []
for j, c in enumerate(cells):
    r = new.get(j, c)
    for c2 in (r if isinstance(r, list) else [r]):
        c2 = copy.deepcopy(c2)
        if c2['cell_type'] == 'code':
            c2['outputs'] = []; c2['execution_count'] = None
        out.append(c2)
nb['cells'] = out
json.dump(nb, open(dst_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('cells', len(out), '| replaced', sorted(new))
