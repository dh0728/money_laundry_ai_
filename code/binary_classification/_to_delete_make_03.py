# 02 노트북을 바탕으로 03_account_activity.ipynb 를 만든다 (출력 제거, 단계 고유 셀 교체)
import json, sys, copy

src_path, dst_path = sys.argv[1], sys.argv[2]
nb = json.load(open(src_path, encoding='utf-8'))
cells = nb['cells']
MD = lambda s: dict(cell_type='markdown', metadata={}, source=s)
CODE = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)

def find(prefix, start=0):
    for i in range(start, len(cells)):
        if ''.join(cells[i]['source']).startswith(prefix):
            return i
    raise KeyError(prefix)

new = {}
new[0] = MD("""# 이진 분류 실험 ③ — 계좌 단기 활동 피처 추가 (account activity)
1단계 피처 사다리의 두 번째 칸. 02(기본 4 + port 2 + pair 4) 위에 **거래 직전 두 계좌의 활동량** 피처 8개를 더한다. 모델은 GINe 고정, 나머지 설정은 01·02 와 동일.

| 피처 | 뜻 (전부 그 거래 시각 기준 과거, 같은 분 제외) |
|---|---|
| `src_out_1h` / `src_out_1d` | 송신 계좌가 최근 60분 / 24시간 안에 보낸 거래 수 |
| `src_uniq_dst_1d` | 송신 계좌가 최근 24시간 안에 보낸 **상대 계좌 수** (일 단위 첫 거래 수로 근사) |
| `src_out_amt_1d` | 송신 계좌가 최근 24시간 안에 보낸 금액 합 |
| `dst_in_1h` / `dst_in_1d` | 수신 계좌가 최근 60분 / 24시간 안에 받은 거래 수 |
| `dst_uniq_src_1d` | 수신 계좌가 최근 24시간 안에 받은 **상대 계좌 수** |
| `dst_in_amt_1d` | 수신 계좌가 최근 24시간 안에 받은 금액 합 |

전부 `ln(1 + x)`. 창은 1h(버스트)와 1d(하루 활동) 두 개 — GFP 논문도 fan/degree 를 시간 창으로 센다 (`sampling/reference.md` 6절).

**가설**: 한 계좌가 짧은 시간에 여러 상대와 거래하는 **FAN-OUT / FAN-IN**, 모았다 뿌리는 **GATHER-SCATTER / SCATTER-GATHER** 의 recall 이 빡빡한 컷(0.1%, argmax)에서 오른다.
02 에서 가장 약했던 **고차수 수신자 [20-49]·[50-99] 구간**의 AP 가 오르는지도 본다 (`dst_in_1d` 가 "지금 이 계좌가 허브인가" 를 직접 말해 줌).

**메모리**: 간선 피처 18열 → FEAT16 이 02 보다 2.9GB 커진다 (02 학습 중 20.7GB → 약 23.6GB 예상, 한도 25GB). 5절에서 RSS 를 확인하고 넘치면 1h 피처 2개를 빼서 다시.

**누수 규칙**: 02 와 같음 — 거래 시각 기준 과거만, 같은 분 제외, z-norm 은 학습 구간 통계. 공통 코드 `bc_lib.py`""")

i = find('# ---- 실험 설정')
new[i] = CODE("""# ---- 실험 설정 ------------------------------------------------------------
RUN_NAME = 'account_activity'     # results/<DATASET>/account_activity/
PREV_RUN = 'pair_history'         # 비교 기준 (02)
cfg = B.Config(dataset='HI-Large', smoke=False)
B.seed_all(cfg.seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = cfg.out_dir(RUN_NAME)

# 앞 단계 피처 (02 에서 만든 parquet 재사용) + 이번 단계 피처
PAIR_FEATURES = ['pair_prior_count', 'pair_gap_minutes', 'pair_count_1h', 'reverse_history_count']
PAIR_PATH = cfg.feat_dir / 'pair_history.parquet'
STAGE_FEATURES = ['src_out_1h', 'src_out_1d', 'src_uniq_dst_1d', 'src_out_amt_1d', 'dst_in_1h', 'dst_in_1d', 'dst_uniq_src_1d', 'dst_in_amt_1d']
STAGE_PATH = cfg.feat_dir / 'account_activity.parquet'
GNN_EDGE_FEATURES = B.BASIC_EDGE_FEATURES + B.PORT_FEATURES + PAIR_FEATURES + STAGE_FEATURES   # FEAT16 열 순서. port 는 4,5 번
TREE_FEATURES = B.BASIC_EDGE_FEATURES + PAIR_FEATURES + STAGE_FEATURES                         # 트리는 port 없이
assert PAIR_PATH.exists(), '02 를 먼저 돌려 pair_history.parquet 을 만들어야 한다'
DEVICE, OUT_DIR, len(GNN_EDGE_FEATURES), len(TREE_FEATURES)""")

i = find('## 1. 데이터 준비')
new[i] = MD("## 1. 데이터 준비 — 거래 전체 + port + pair (02 parquet)")
i = find('con = L.connect')
new[i] = CODE("""con = L.connect(threads=8, memory='12GB')
currency_codes, format_codes = B.code_tables(con, cfg)
PORTS_PATH = B.ensure_ports(con, cfg)               # 있으면 건너뜀
tx = B.load_tx(con, cfg)
tx = B.attach_parquet(tx, PORTS_PATH, B.PORT_FEATURES)
tx = B.attach_parquet(tx, PAIR_PATH, PAIR_FEATURES)   # 02 의 pair 피처 (edge_id 순 parquet → 열 붙이기)
tx.shape""")

i = find('## 2. 이번 단계 피처 계산')
new[i] = MD("## 2. 이번 단계 피처 계산 — 계좌 단기 활동")
new[i + 1] = MD("""- DuckDB 창 함수 두 번: **송신 계좌(src) 파티션** 한 번, **수신 계좌(dst) 파티션** 한 번. 02 의 pair 창과 같은 꼴 (`RANGE … PRECEDING AND INTERVAL '1 minute' PRECEDING`)
- 상대 계좌 수(`uniq_*_1d`): 창 함수로 `count(DISTINCT)` 를 1.8억 건에 돌리면 무거워서, **(계좌, 상대, 일) 의 첫 거래에 1 을 찍고 24시간 창에서 합산**으로 근사. 같은 상대가 어제 밤·오늘 아침 둘 다 있으면 2로 세는 오차가 있지만 방향은 같다
- 금액은 `received`(Amount Received, 기본 피처와 같은 열) 의 창 합계
- 결과는 `features_bc/account_activity.parquet` (edge_id 순). 한 번 만들면 재사용. pandas 를 거치지 않는다""")
new[i + 2] = CODE("""if not STAGE_PATH.exists():
    t0 = time.time()
    con.execute(f\"\"\"
        CREATE OR REPLACE TABLE tx_min AS
        SELECT edge_id, src, dst, ts, day, received FROM read_parquet({L.q(cfg.tx_path)})\"\"\")
    # (a) 송신 계좌 기준 — 같은 src 의 과거 보낸 거래. first_pair_day = (src, dst, 일) 안의 첫 거래 표시 (상대 수 근사용)
    con.execute(\"\"\"
        CREATE OR REPLACE TABLE src_win AS
        SELECT edge_id,
               count(*)            OVER w1h AS out_1h,
               count(*)            OVER w1d AS out_1d,
               sum(first_pair_day) OVER w1d AS uniq_dst_1d,
               sum(received)       OVER w1d AS out_amt_1d
        FROM (SELECT *, CASE WHEN row_number() OVER (PARTITION BY src, dst, day ORDER BY ts, edge_id) = 1 THEN 1 ELSE 0 END AS first_pair_day FROM tx_min)
        WINDOW w1h AS (PARTITION BY src ORDER BY ts RANGE BETWEEN INTERVAL '60 minutes' PRECEDING AND INTERVAL '1 minute' PRECEDING),
               w1d AS (PARTITION BY src ORDER BY ts RANGE BETWEEN INTERVAL '24 hours'  PRECEDING AND INTERVAL '1 minute' PRECEDING)\"\"\")
    print(f'  src_win {time.time()-t0:.0f}s')
    # (b) 수신 계좌 기준 — 같은 dst 의 과거 받은 거래
    con.execute(\"\"\"
        CREATE OR REPLACE TABLE dst_win AS
        SELECT edge_id,
               count(*)            OVER w1h AS in_1h,
               count(*)            OVER w1d AS in_1d,
               sum(first_pair_day) OVER w1d AS uniq_src_1d,
               sum(received)       OVER w1d AS in_amt_1d
        FROM (SELECT *, CASE WHEN row_number() OVER (PARTITION BY dst, src, day ORDER BY ts, edge_id) = 1 THEN 1 ELSE 0 END AS first_pair_day FROM tx_min)
        WINDOW w1h AS (PARTITION BY dst ORDER BY ts RANGE BETWEEN INTERVAL '60 minutes' PRECEDING AND INTERVAL '1 minute' PRECEDING),
               w1d AS (PARTITION BY dst ORDER BY ts RANGE BETWEEN INTERVAL '24 hours'  PRECEDING AND INTERVAL '1 minute' PRECEDING)\"\"\")
    print(f'  dst_win {time.time()-t0:.0f}s')
    # (c) ln 변환 후 저장 (edge_id 순)
    con.execute(f\"\"\"
        COPY (
            SELECT t.edge_id,
                   ln(1 + s.out_1h)::FLOAT                   AS src_out_1h,
                   ln(1 + s.out_1d)::FLOAT                   AS src_out_1d,
                   ln(1 + coalesce(s.uniq_dst_1d, 0))::FLOAT AS src_uniq_dst_1d,
                   ln(1 + coalesce(s.out_amt_1d, 0))::FLOAT  AS src_out_amt_1d,
                   ln(1 + d.in_1h)::FLOAT                    AS dst_in_1h,
                   ln(1 + d.in_1d)::FLOAT                    AS dst_in_1d,
                   ln(1 + coalesce(d.uniq_src_1d, 0))::FLOAT AS dst_uniq_src_1d,
                   ln(1 + coalesce(d.in_amt_1d, 0))::FLOAT   AS dst_in_amt_1d
            FROM tx_min t JOIN src_win s USING (edge_id) JOIN dst_win d USING (edge_id)
            ORDER BY t.edge_id
        ) TO {L.q(STAGE_PATH)} (FORMAT PARQUET, COMPRESSION ZSTD)\"\"\")
    con.execute("DROP TABLE src_win; DROP TABLE dst_win; DROP TABLE tx_min")
    print(f'{STAGE_PATH.name} 생성 {time.time()-t0:.0f}s')
con.execute(f"SELECT count(*), {', '.join(f'avg({c}) AS avg_{c}' for c in STAGE_FEATURES)} FROM read_parquet({L.q(STAGE_PATH)})").df().T""")
new[i + 3] = CODE("""# 창 동작 확인 — 하루에 거래 8건쯤 보낸 계좌 하나를 골라 그 계좌의 보낸 거래 타임라인 (out_1h 는 1시간 지나면 줄고, out_1d 는 하루 안에서 1씩 늘고, uniq 는 새 상대일 때만 느는지)
# 주의: 1.8억 건끼리 조인하지 않는다 → 그 계좌의 edge_id 만 골라 피처 parquet 에서 읽는다
acct = con.execute(f"SELECT src FROM read_parquet({L.q(cfg.tx_path)}) GROUP BY src, day ORDER BY abs(count(*) - 8), src LIMIT 1").fetchone()[0]
ids = con.execute(f"SELECT edge_id, ts, dst, received FROM read_parquet({L.q(cfg.tx_path)}) WHERE src = {acct} ORDER BY ts LIMIT 40").df()
feat = con.execute(f"SELECT edge_id, src_out_1h, src_out_1d, src_uniq_dst_1d, src_out_amt_1d FROM read_parquet({L.q(STAGE_PATH)}) WHERE edge_id IN ({', '.join(map(str, ids.edge_id))})").df()
tl = ids.merge(feat, on='edge_id').sort_values('ts')
for c in ['src_out_1h', 'src_out_1d', 'src_uniq_dst_1d']:
    tl[c.replace('src_', '') + ' (건수)'] = np.expm1(tl[c]).round(0).astype(int)      # ln(1+x) 를 되돌려 건수로
tl""")

i = find('# 트리 모델')
new[i] = MD("# 트리 모델 (기본 4 + pair 4 + 이번 피처 8)")

i = find('# 배치 하나 꺼내서')
new[i] = CODE(''.join(cells[i]['source']).replace('엣지 피처 10열', '엣지 피처 18열'))

i = find('store = B.GraphStore')
new[i] = CODE("""store = B.GraphStore(tx, feat_z, cfg, train_rows=np.flatnonzero(train_mask), val_idx=val_idx, test_idx=test_idx,
                     port_cols=[GNN_EDGE_FEATURES.index('in_port'), GNN_EDGE_FEATURES.index('out_port')])
del tx, feat_z; B.trim('(tx, feat_z 삭제 후)')
# 02 는 이 시점 RSS 가 약 13GB, 학습 그래프까지 올리면 20.7GB 였다. 여기서 16GB 를 넘으면 학습 중 25GB 한도를 넘길 수 있다 → STAGE_FEATURES 에서 1h 두 개를 빼고 다시
store.edge_dim""")

i = find('## 14. 다음')
new[i] = MD("""## 14. 다음
- 04: 계좌 금액 분포 (1일 창 mean/std/max, 이 거래 금액이 그 계좌 평소 대비 몇 배) → 스머핑·허브 구별
- 05: GFP 모티프 (사이클·scatter-gather 참여 수, snapml GraphFeaturePreprocessor) → CYCLE, SCATTER-GATHER
- 그 다음 v10 36피처 참조 런 → 1단계 정리""")

drop = set()
for prefix in ("results['gine'] = json.load(open(OUT_DIR / 'results.json'))", "p = lgb_model.predict_proba(X_val)"):
    try:
        drop.add(find(prefix))
    except KeyError:
        pass

out = []
for j, c in enumerate(cells):
    if j in drop:
        continue
    c = copy.deepcopy(new.get(j, c))
    if c['cell_type'] == 'code':
        c['outputs'] = []; c['execution_count'] = None
    out.append(c)
nb['cells'] = out
json.dump(nb, open(dst_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('cells', len(out), '| dropped', sorted(drop), '| replaced', sorted(new))
