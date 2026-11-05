"""v10 daily-close 피처 재현 — 공통 모듈.

팀원 SALT-GNN 노트북(00~05, salt_daily/raw_common.py · preprocessing.py)의 전처리를
우리 code/01_prepare_parquet 출력(transactions / kyc / patterns.parquet)에서 같은 정의로 다시 만든다.
노트북: code/nonpaten_vs_normal/04~06, 출력: parquets/<DATASET>/nonpaten_vs_normal/v10_lgbm/
GNN 과 같은 입력으로 LightGBM 을 돌려 "피처에 무패턴 신호가 있나" 를 가르는 용도.

팀원 코드와 같게 둔 것 (바꾸면 비교가 깨짐)
  - 이력은 2022-08-01 부터 누적. 모든 피처가 그 거래 이전(노드는 그날 마감) 거래만 쓰므로 관측 끝을 어디에 두든 앞쪽 값은 같다
  - 엣지 17개: 같은 분 제외 창(pair/sender/receiver), 어제까지 쌍·역방향 이력, 8/1~8/7 로 맞춘 통화별 금액 z
  - 결제유형 코드 0~7, port (쌍의 첫 거래 순서)
  - 노드 19개: 그날 마감(당일 거래 포함)까지 누적한 일별 스냅샷. 거래가 없는 계좌는 0

팀원 코드와 다를 수 있는 것
  - edge_id / node_id 번호 체계 (우리 code/01_prepare_parquet 의 정렬 순서). 값에는 영향 없고 port 의 같은 분 동률 순서만 바뀔 수 있음
  - 금액은 우리 parquet 의 DOUBLE (팀원은 DECIMAL(38,8) → DOUBLE). 마지막 자리 반올림 차이 수준
  - 분할은 우리 binary_classification 실험 것 (Train 8/8~10/2, Val 10/3~10/16, Test 10/17~10/30, 10/31·11월 미사용)
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATASET = "HI-Large"
OBS_START, OBS_END = "2022-08-01", "2022-10-31"          # 관측 구간 [start, end) — 분할 끝 10/30 까지
CALIB_END = "2022-08-08"                                  # 금액 보정 = 8/1~8/7
SPLITS = {                                                # 양끝 포함 — binary_classification(sampling_lib) 분할
    "train": ("2022-08-08", "2022-10-02"),
    "val": ("2022-10-03", "2022-10-16"),
    "test": ("2022-10-17", "2022-10-30"),
}
# binary_classification 04 노트북에 찍힌 구간별 거래 수 (HI-Large). 같으면 같은 모집단
EXPECTED_ROWS = {"train": 102_336_629, "val": 24_413_046, "test": 26_808_230}

EDGE_FEATURES = ["pair_prior_count", "pair_gap_minutes", "pair_count_1h", "sender_gap_minutes", "sender_count_1h",
                 "receiver_gap_minutes", "receiver_count_1h", "paid_currency_z", "pair_history_count",
                 "reverse_history_count", "pair_age_at_day_start", "same_entity", "same_known_country",
                 "both_countries_known", "same_currency", "is_self_transfer", "amt_z_pay_ccy"]
# GNN 이 실제로 쓰는 14개 (data.py 의 excluded = amt_z_pay_ccy, pair_history_count, same_entity)
GNN_EDGE_FEATURES = [f for f in EDGE_FEATURES if f not in ("amt_z_pay_ccy", "pair_history_count", "same_entity")]
NODE_FEATURES = ["log_out_count", "log_in_count", "log_out_neighbors", "log_in_neighbors", "log_self_count",
                 "log_reciprocal_neighbors", "mean_paid_currency_z", "paid_outlier_fraction", "log_currency_count",
                 "log_active_days", "type_Corporation", "type_Partnership", "type_Sole Proprietorship",
                 "type_Individual", "type_Country", "type_Direct", "log_entity_observed_accounts",
                 "log_entity_out_diversity", "log_entity_in_diversity"]
ENTITY_TYPES = ["Corporation", "Partnership", "Sole Proprietorship", "Individual", "Country", "Direct"]
FORMAT_NAMES = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire", "Unknown"]
PATTERN_NAMES = ["NORMAL", "BIPARTITE", "CYCLE", "FAN-IN", "FAN-OUT", "GATHER-SCATTER", "RANDOM",
                 "SCATTER-GATHER", "STACK", "UNMATCHED"]          # cls 코드 0~9 (팀원 positive_classes 와 같은 번호)
assert len(EDGE_FEATURES) == 17 and len(GNN_EDGE_FEATURES) == 14 and len(NODE_FEATURES) == 19


# ============================================================================ 경로·유틸
class Paths:
    def __init__(self, dataset=DATASET, root=None, out_name="nonpaten_vs_normal/v10_lgbm"):
        if root is None:
            here = Path.cwd().resolve()
            root = next((p for p in [here, *here.parents]
                         if (p / "parquets" / dataset / "transactions.parquet").is_file()), None)
            if root is None:
                raise FileNotFoundError(f"parquets/{dataset}/transactions.parquet 를 못 찾음 — 01_prepare_parquet 먼저")
        self.root = Path(root).resolve()
        self.base = self.root / "parquets" / dataset
        self.tx, self.kyc, self.patterns = (self.base / f"{n}.parquet" for n in ("transactions", "kyc", "patterns"))
        self.out = self.base / out_name
        self.edge = self.out / "edge_features.parquet"      # edge_id 순, 관측 구간 전체
        self.meta = self.out / "meta.json"                   # 단계별 기록
        self.nodes = self.out / "nodes"                      # <YYYY-MM-DD>.npy (계좌 수 × 19, float32)
        self.first = self.out / "first_days"                 # 노드 단계 보조표 (쌍·상호·소유주 첫 거래일)
        self.lgbm = self.out / "lgbm"                        # <split>.parquet
        self.tmp = self.out / "duckdb_tmp"
        self.out.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"Paths(base={self.base}, out={self.out})"


def q(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def log(msg):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def connect(paths: Paths, threads=4, memory="8GB", db_file=None):
    """파일 DB + temp_directory → 메모리 한도를 넘는 정렬·창 함수는 디스크로 스필한다."""
    paths.tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_file) if db_file else ":memory:")
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit={q(memory)}")
    con.execute(f"SET temp_directory={q(paths.tmp)}")
    con.execute("SET preserve_insertion_order=false")
    return con


def read_meta(paths):
    return json.loads(paths.meta.read_text(encoding="utf-8")) if paths.meta.exists() else {}


def write_meta(paths, key, value):
    meta = read_meta(paths); meta[key] = value
    paths.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def day_index(day) -> int:
    """2022-08-01 을 0 으로 한 날짜 번호 (팀원 코드의 day)."""
    return (pd.Timestamp(day).date() - date.fromisoformat(OBS_START)).days


def days_between(a, b):
    """a~b (양끝 포함) 날짜 문자열 목록."""
    d, e, out = date.fromisoformat(a), date.fromisoformat(b), []
    while d <= e:
        out.append(str(d)); d += timedelta(days=1)
    return out


# ============================================================================ 1. 입력 뷰
def register_inputs(con, paths):
    """obs  : 관측 구간 거래 (중복 대표만). 라벨 열은 넣지 않는다 → 피처 계산이 라벨을 볼 수 없음
       kyc_t: 계좌별 소유주 / entity_type / country — 팀원 raw_common.ingest 와 같은 정규식"""
    con.execute(f"""
        CREATE OR REPLACE VIEW obs AS
        SELECT edge_id, ts, src, dst, paid, received, payment_currency, receiving_currency, payment_format,
               CAST(ts AS DATE) AS day
        FROM read_parquet({q(paths.tx)})
        WHERE (duplicate_group_id IS NULL OR edge_id = duplicate_group_id)
          AND ts >= TIMESTAMP {q(OBS_START)} AND ts < TIMESTAMP {q(OBS_END)}""")
    con.execute(f"""
        CREATE OR REPLACE TABLE kyc_t AS
        SELECT node_id, entity_id,
               coalesce(regexp_extract(entity_name, '^([A-Za-z ]+?)\\s*#', 1), '') AS entity_type,
               CASE WHEN lower(bank_name) LIKE '%crypto%' OR lower(bank_name) LIKE '%crytpo%' THEN ''
                    ELSE coalesce(regexp_extract(bank_name, '^(.*?) Bank #', 1), '') END AS country
        FROM read_parquet({q(paths.kyc)})""")


def split_counts(con):
    """구간별 관측 거래 수 (라벨 무관). EXPECTED_ROWS(binary_classification 실험) 와 같아야 같은 모집단."""
    rows = []
    for name, (a, b) in SPLITS.items():
        n = con.execute(f"SELECT count(*) FROM obs WHERE day BETWEEN DATE {q(a)} AND DATE {q(b)}").fetchone()[0]
        rows.append(dict(split=name, start=a, end=b, rows=n, expected=EXPECTED_ROWS[name], same=n == EXPECTED_ROWS[name]))
    return pd.DataFrame(rows)


def format_case(col):
    return "CASE " + col + " " + " ".join(f"WHEN {q(n)} THEN {i}" for i, n in enumerate(FORMAT_NAMES[:-1])) + " ELSE 7 END"


def currency_list(con):
    """노드 피처의 통화 비트마스크 순서 — 관측 구간의 지불·수취 통화 합집합을 정렬 (팀원과 같음)."""
    return [r[0] for r in con.execute(
        "SELECT payment_currency FROM obs UNION SELECT receiving_currency FROM obs ORDER BY 1").fetchall()]


# ============================================================================ 2. 엣지 피처
def build_edge_steps(con):
    """팀원 build_transaction_features + build_edges 의 SQL 을 단계별로. (이름, SQL) 목록을 돌려준다.
    노트북에서 단계마다 시간을 찍으려고 나눠 둔다."""
    gap = lambda name, end="t.ts": f"CASE WHEN {name} IS NULL THEN 0 ELSE ln(2 + date_diff('minute', {name}, {end})) END"
    steps = [
        ("calibration: 8/1~8/7 의 지불액(지불통화)·수취액(수취통화)", f"""
            CREATE OR REPLACE TABLE calibration AS
            SELECT payment_currency AS currency, paid::DOUBLE AS amount FROM obs WHERE ts < TIMESTAMP {q(CALIB_END)}
            UNION ALL
            SELECT receiving_currency, received::DOUBLE FROM obs WHERE ts < TIMESTAMP {q(CALIB_END)}"""),
        ("robust: 통화별 scale(양수 중앙값) → ln(1+금액/scale) 의 중앙값·IQR", """
            CREATE OR REPLACE TABLE scales AS
            SELECT currency, coalesce(median(amount) FILTER (WHERE amount > 0), 1.) AS scale FROM calibration GROUP BY currency;
            CREATE OR REPLACE TABLE robust AS
            SELECT currency, scale, median(z) AS med, greatest(quantile_cont(z, .75) - quantile_cont(z, .25), .001) AS iqr
            FROM (SELECT currency, scale, ln(1 + amount / scale) AS z FROM calibration JOIN scales USING (currency))
            GROUP BY currency, scale"""),
        ("amount_stats: 통화별 ln(1+지불액) 평균·모표준편차 (amt_z_pay_ccy 용)", f"""
            CREATE OR REPLACE TABLE amount_stats AS
            SELECT payment_currency, avg(ln(1 + paid::DOUBLE)) AS m, stddev_pop(ln(1 + paid::DOUBLE)) AS sd
            FROM obs WHERE day < DATE {q(CALIB_END)} GROUP BY payment_currency"""),
        ("windows: 같은 분 제외 과거 창 (쌍 누적·쌍 1h·송신/수신 마지막 시각·1h)", """
            CREATE OR REPLACE TABLE windows AS
            SELECT edge_id,
                   count(*) OVER pair_all AS pair_count, max(ts) OVER pair_all AS pair_last,
                   count(*) OVER pair_hour AS pair_hour_count,
                   max(ts) OVER sender_all AS sender_last, count(*) OVER sender_hour AS sender_hour_count,
                   max(ts) OVER receiver_all AS receiver_last, count(*) OVER receiver_hour AS receiver_hour_count
            FROM obs WINDOW
              pair_all      AS (PARTITION BY src, dst ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 MINUTE PRECEDING),
              pair_hour     AS (PARTITION BY src, dst ORDER BY ts RANGE BETWEEN INTERVAL 60 MINUTE PRECEDING AND INTERVAL 1 MINUTE PRECEDING),
              sender_all    AS (PARTITION BY src ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 MINUTE PRECEDING),
              sender_hour   AS (PARTITION BY src ORDER BY ts RANGE BETWEEN INTERVAL 60 MINUTE PRECEDING AND INTERVAL 1 MINUTE PRECEDING),
              receiver_all  AS (PARTITION BY dst ORDER BY ts RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL 1 MINUTE PRECEDING),
              receiver_hour AS (PARTITION BY dst ORDER BY ts RANGE BETWEEN INTERVAL 60 MINUTE PRECEDING AND INTERVAL 1 MINUTE PRECEDING)"""),
        ("history: 어제까지 같은 쌍 누적·마지막 시각, 역방향 누적 (ASOF)", """
            CREATE OR REPLACE TABLE pair_days AS
            SELECT src, dst, day, max(ts) AS last_ts, sum(count(*)) OVER (PARTITION BY src, dst ORDER BY day) AS cumulative
            FROM obs GROUP BY src, dst, day;
            CREATE OR REPLACE TABLE history AS
            SELECT t.edge_id, p.last_ts AS pair_last_day, coalesce(p.cumulative, 0) AS pair_cumulative,
                   coalesce(r.cumulative, 0) AS reverse_count
            FROM obs t
            ASOF LEFT JOIN pair_days p ON t.src = p.src AND t.dst = p.dst AND t.day > p.day
            ASOF LEFT JOIN pair_days r ON t.src = r.dst AND t.dst = r.src AND t.day > r.day"""),
        ("ports: 쌍의 첫 거래 순서 (수신 계좌 기준 in_port, 송신 계좌 기준 out_port)", """
            CREATE OR REPLACE TABLE first_pair AS
            SELECT src, dst, min(ts) AS first_ts, arg_min(edge_id, (ts, edge_id)) AS first_id FROM obs GROUP BY src, dst;
            CREATE OR REPLACE TABLE ports AS
            SELECT src, dst,
                   row_number() OVER (PARTITION BY dst ORDER BY first_ts, first_id) - 1 AS in_port,
                   row_number() OVER (PARTITION BY src ORDER BY first_ts, first_id) - 1 AS out_port
            FROM first_pair"""),
    ]
    expressions = [
        "ln(1 + w.pair_count)", gap("w.pair_last"), "ln(1 + w.pair_hour_count)",
        gap("w.sender_last"), "ln(1 + w.sender_hour_count)",
        gap("w.receiver_last"), "ln(1 + w.receiver_hour_count)",
        "CASE WHEN r.currency IS NULL THEN 0 ELSE greatest(-8, least(8, (ln(1 + t.paid::DOUBLE / r.scale) - r.med) / r.iqr)) END",
        "ln(1 + h.pair_cumulative)", "ln(1 + h.reverse_count)", gap("h.pair_last_day", "CAST(t.day AS TIMESTAMP)"),
        "(s.entity_id IS NOT NULL AND s.entity_id = d.entity_id)::INTEGER",
        "(s.country <> '' AND d.country <> '' AND s.country = d.country)::INTEGER",
        "(s.country <> '' AND d.country <> '')::INTEGER",
        "(t.payment_currency = t.receiving_currency)::INTEGER", "(t.src = t.dst)::INTEGER",
        "CASE WHEN a.sd IS NULL THEN 0 ELSE greatest(-10, least(10, (ln(1 + t.paid::DOUBLE) - a.m) / (a.sd + 0.000001))) END",
    ]
    cols = ", ".join(f'({e})::FLOAT AS "{n}"' for n, e in zip(EDGE_FEATURES, expressions))
    final = f"""
        CREATE OR REPLACE TABLE edge_out AS
        SELECT t.edge_id, t.day, date_diff('minute', TIMESTAMP {q(OBS_START)}, t.ts)::INTEGER AS minute,
               t.src::INTEGER AS src, t.dst::INTEGER AS dst,
               ({format_case('t.payment_format')})::TINYINT AS payment_format_code,
               {cols},
               p.in_port::INTEGER AS in_port, p.out_port::INTEGER AS out_port,
               t.payment_currency, t.receiving_currency
        FROM obs t JOIN windows w USING (edge_id) JOIN history h USING (edge_id)
        JOIN ports p ON p.src = t.src AND p.dst = t.dst
        LEFT JOIN robust r ON t.payment_currency = r.currency
        LEFT JOIN amount_stats a ON a.payment_currency = t.payment_currency
        JOIN kyc_t s ON s.node_id = t.src JOIN kyc_t d ON d.node_id = t.dst"""
    steps.append(("edge_out: 17개 조립 + 결제유형 코드 + port", final))
    return steps


def export_edges(con, paths):
    con.execute(f"COPY (SELECT * FROM edge_out ORDER BY edge_id) TO {q(paths.edge)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)")


# ============================================================================ 3. 노드 피처 (일별 누적 스냅샷)
def popcount64(a):
    a = np.ascontiguousarray(a, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a).astype(np.int64)
    return np.unpackbits(a.view(np.uint8).reshape(-1, 8), axis=1).sum(1).astype(np.int64)


def build_first_days(con, paths):
    """팀원 build_daily_nodes 의 보조표 3개를 first_day 순 parquet 으로 (하루치만 골라 읽기 쉽게).
    pair_first      : 쌍(src,dst) 의 첫 거래일 → 그날 out/in 이웃 수 +1
    reciprocal_first: a→b, b→a 가 둘 다 생긴 날(늦은 쪽) → 그날 a 의 상호 이웃 수 +1
    entity_first    : 소유주 쌍의 첫 거래일 → 소유주 out/in 다양성 +1"""
    paths.first.mkdir(parents=True, exist_ok=True)
    con.execute("""
        CREATE OR REPLACE TABLE owners AS
        WITH o AS (SELECT entity_id, (row_number() OVER (ORDER BY entity_id) - 1)::INTEGER AS owner_code
                   FROM (SELECT DISTINCT entity_id FROM kyc_t WHERE entity_id IS NOT NULL))
        SELECT k.node_id, coalesce(o.owner_code, -1) AS owner_code, k.entity_type
        FROM kyc_t k LEFT JOIN o USING (entity_id)""")
    con.execute("CREATE OR REPLACE TABLE pair_first AS SELECT src, dst, min(day) AS first_day FROM obs GROUP BY src, dst")
    con.execute(f"COPY (SELECT * FROM pair_first ORDER BY first_day) TO {q(paths.first / 'pair_first.parquet')} (FORMAT PARQUET)")
    con.execute(f"""COPY (SELECT a.src AS node, greatest(a.first_day, b.first_day)::DATE AS first_day, count(*) AS n
                        FROM pair_first a JOIN pair_first b ON a.src = b.dst AND a.dst = b.src WHERE a.src <> a.dst
                        GROUP BY 1, 2 ORDER BY first_day) TO {q(paths.first / 'reciprocal_first.parquet')} (FORMAT PARQUET)""")
    con.execute(f"""COPY (SELECT a.owner_code AS source_owner, b.owner_code AS target_owner, min(p.first_day)::DATE AS first_day
                        FROM pair_first p JOIN owners a ON a.node_id = p.src JOIN owners b ON b.node_id = p.dst
                        WHERE a.owner_code >= 0 AND b.owner_code >= 0 GROUP BY 1, 2 ORDER BY first_day)
                   TO {q(paths.first / 'entity_first.parquet')} (FORMAT PARQUET)""")
    con.execute("DROP TABLE pair_first")


def _day_rows(con, path, day, cols, key="first_day"):
    df = con.execute(f"SELECT {', '.join(cols)} FROM read_parquet({q(path)}) WHERE {key} = DATE {q(day)}").fetchnumpy()
    return {c: np.asarray(df[c]) for c in cols}


def build_daily_nodes(con, paths, save_days=None, verbose=True):
    """팀원 build_daily_nodes 와 같은 누적 계산. 8/1 부터 하루씩 더하며 그날 마감 스냅샷(계좌 수 × 19)을 저장한다.
    save_days: 저장할 날짜 집합 (None 이면 전부). 누적은 항상 8/1 부터 한다."""
    paths.nodes.mkdir(parents=True, exist_ok=True)
    k = con.execute("SELECT node_id, owner_code, entity_type FROM owners ORDER BY node_id").df()
    n = int(con.execute(f"SELECT count(*) FROM read_parquet({q(paths.kyc)})").fetchone()[0])
    assert len(k) == n and (k.node_id.to_numpy() == np.arange(n)).all(), "kyc.node_id 가 0..N-1 이 아님"
    owner = k.owner_code.to_numpy(np.int64)
    types = k.entity_type.map({v: i for i, v in enumerate(ENTITY_TYPES)}).fillna(-1).to_numpy(np.int8)
    ne = int(owner.max()) + 1
    currencies = currency_list(con)
    if len(currencies) > 64:
        raise ValueError("통화가 64개를 넘음")
    code = {c: i for i, c in enumerate(currencies)}

    counts = np.zeros((n, 6), np.int64); zsum = np.zeros(n); tails = np.zeros(n, np.int64)
    active_days = np.zeros(n, np.int16); currency = np.zeros(n, np.uint64)
    owner_n = np.zeros(ne, np.int64); owner_out = np.zeros(ne, np.int64); owner_in = np.zeros(ne, np.int64)
    first = paths.first
    total = 0
    for ds in days_between(OBS_START, str(date.fromisoformat(OBS_END) - timedelta(days=1))):
        r = con.execute(f"""SELECT src, dst, paid_currency_z, payment_currency, receiving_currency
                            FROM read_parquet({q(paths.edge)}) WHERE day = DATE {q(ds)}""").df()
        s, d = r.src.to_numpy(np.int64), r.dst.to_numpy(np.int64); z = r.paid_currency_z.to_numpy(np.float64)
        pc = r.payment_currency.map(code).to_numpy(np.uint64); rc = r.receiving_currency.map(code).to_numpy(np.uint64)
        total += len(s)
        counts[:, 0] += np.bincount(s, minlength=n); counts[:, 1] += np.bincount(d, minlength=n)
        counts[:, 4] += np.bincount(s[s == d], minlength=n)
        zsum += np.bincount(s, weights=z, minlength=n); tails += np.bincount(s[np.abs(z) >= 3], minlength=n)
        np.bitwise_or.at(currency, s, np.left_shift(np.uint64(1), pc))
        np.bitwise_or.at(currency, d, np.left_shift(np.uint64(1), rc))
        active = np.union1d(s, d); new = active[active_days[active] == 0]; own = owner[new]; own = own[own >= 0]
        owner_n += np.bincount(own, minlength=ne); active_days[active] += 1
        pair = _day_rows(con, first / "pair_first.parquet", ds, ["src", "dst"])
        counts[:, 2] += np.bincount(pair["src"].astype(np.int64), minlength=n)
        counts[:, 3] += np.bincount(pair["dst"].astype(np.int64), minlength=n)
        rec = _day_rows(con, first / "reciprocal_first.parquet", ds, ["node", "n"])
        np.add.at(counts[:, 5], rec["node"].astype(np.int64), rec["n"].astype(np.int64))
        ep = _day_rows(con, first / "entity_first.parquet", ds, ["source_owner", "target_owner"])
        owner_out += np.bincount(ep["source_owner"].astype(np.int64), minlength=ne)
        owner_in += np.bincount(ep["target_owner"].astype(np.int64), minlength=ne)
        if save_days is None or ds in save_days:
            x = np.zeros((n, 19), np.float32); x[:, :6] = np.log1p(counts)
            x[:, 6] = np.divide(zsum, counts[:, 0], out=np.zeros(n), where=counts[:, 0] > 0)
            x[:, 7] = np.divide(tails, counts[:, 0], out=np.zeros(n), where=counts[:, 0] > 0)
            x[:, 8] = np.log1p(popcount64(currency).astype(float)); x[:, 9] = np.log1p(active_days.astype(float))
            for j in range(6):
                x[:, 10 + j] = types == j
            valid = owner >= 0
            for j, values in [(16, owner_n), (17, owner_out), (18, owner_in)]:
                x[valid, j] = np.log1p(values[owner[valid]])
            x[active_days == 0] = 0
            assert np.isfinite(x).all()
            tmp = paths.nodes / f".{ds}.tmp.npy"
            np.save(tmp, x); tmp.replace(paths.nodes / f"{ds}.npy")
            del x
        if verbose:
            log(f"nodes {ds}  거래 {len(s):,}  활동 계좌 누적 {(active_days > 0).sum():,}")
    return dict(nodes=n, rows=total, currencies=currencies, owners=ne)


def node_snapshot_sql(con, paths, end_day, node_ids):
    """검증용: 그날 마감까지의 거래를 SQL 로 한 번에 집계한 19개 (팀원 raw_common.build_node_snapshot 과 같은 식).
    node_ids 행만 돌려준다 (활동 없는 계좌는 0)."""
    end = str(date.fromisoformat(end_day) + timedelta(days=1))
    con.register("want_nodes", pd.DataFrame({"node_id": np.asarray(node_ids, np.int64)}))
    one_hot = ",".join(f"(k.entity_type={q(n)})::INTEGER" for n in ENTITY_TYPES)
    df = con.execute(f"""
        WITH prefix AS (SELECT src, dst, day, payment_currency, receiving_currency, paid_currency_z AS paid_z
                        FROM read_parquet({q(paths.edge)}) WHERE day < DATE {q(end)}),
        pairs AS (SELECT DISTINCT src, dst FROM prefix),
        out_stats AS (SELECT src node_id, count(*) nout, count(DISTINCT dst) peers, count(*) FILTER (WHERE src = dst) self_count,
                             avg(paid_z) z, avg((abs(paid_z) >= 3)::INTEGER) tail FROM prefix GROUP BY src),
        in_stats AS (SELECT dst node_id, count(*) nin, count(DISTINCT src) peers FROM prefix GROUP BY dst),
        reciprocal AS (SELECT a.src node_id, count(*) n FROM pairs a JOIN pairs b ON a.src = b.dst AND a.dst = b.src
                       WHERE a.src <> a.dst GROUP BY a.src),
        currency AS (SELECT node_id, count(DISTINCT currency) n FROM (SELECT src node_id, payment_currency currency FROM prefix
                     UNION ALL SELECT dst, receiving_currency FROM prefix) GROUP BY node_id),
        active AS (SELECT node_id, count(DISTINCT day) n FROM (SELECT src node_id, day FROM prefix UNION ALL SELECT dst, day FROM prefix) GROUP BY node_id),
        owner_nodes AS (SELECT k.entity_id, count(*) n FROM kyc_t k JOIN active a USING (node_id) WHERE k.entity_id IS NOT NULL GROUP BY k.entity_id),
        owner_pairs AS (SELECT DISTINCT s.entity_id source_owner, d.entity_id target_owner FROM pairs p
                        JOIN kyc_t s ON p.src = s.node_id JOIN kyc_t d ON p.dst = d.node_id
                        WHERE s.entity_id IS NOT NULL AND d.entity_id IS NOT NULL),
        owner_out AS (SELECT source_owner entity_id, count(*) n FROM owner_pairs GROUP BY 1),
        owner_in AS (SELECT target_owner entity_id, count(*) n FROM owner_pairs GROUP BY 1)
        SELECT w.node_id, ln(1+coalesce(o.nout,0)), ln(1+coalesce(i.nin,0)), ln(1+coalesce(o.peers,0)), ln(1+coalesce(i.peers,0)),
               ln(1+coalesce(o.self_count,0)), ln(1+coalesce(r.n,0)), coalesce(o.z,0), coalesce(o.tail,0), ln(1+c.n), ln(1+a.n),
               {one_hot}, ln(1+coalesce(own.n,0)), ln(1+coalesce(oo.n,0)), ln(1+coalesce(oi.n,0))
        FROM want_nodes w JOIN kyc_t k USING (node_id) JOIN active a USING (node_id)
        LEFT JOIN out_stats o USING (node_id) LEFT JOIN in_stats i USING (node_id) LEFT JOIN reciprocal r USING (node_id)
        JOIN currency c USING (node_id) LEFT JOIN owner_nodes own USING (entity_id)
        LEFT JOIN owner_out oo USING (entity_id) LEFT JOIN owner_in oi USING (entity_id)""").fetchnumpy()
    con.unregister("want_nodes")
    cols = list(df)
    out = np.zeros((len(node_ids), 19), np.float64)
    pos = {int(v): i for i, v in enumerate(node_ids)}
    rows = np.array([pos[int(v)] for v in df[cols[0]]], dtype=np.int64)
    if len(rows):
        out[rows] = np.column_stack([np.asarray(df[c], np.float64) for c in cols[1:]])
    return out


# ============================================================================ 4. LightGBM 입력
LABEL_COLUMNS = ["edge_id", "day", "y", "cls", "w"]
MODEL_COLUMNS = (EDGE_FEATURES + ["payment_format_code", "in_port", "out_port"]
                 + [f"src_{f}" for f in NODE_FEATURES] + [f"dst_{f}" for f in NODE_FEATURES])
# GNN 과 같은 정보량 (엣지 14 + 결제유형 + port 2 + 양 끝 노드 19×2)
GNN_PARITY_COLUMNS = (GNN_EDGE_FEATURES + ["payment_format_code", "in_port", "out_port"]
                      + [f"src_{f}" for f in NODE_FEATURES] + [f"dst_{f}" for f in NODE_FEATURES])


def register_labels(con, paths):
    """labels: edge_id, y(0/1), cls(0 정상 / 1~8 패턴 / 9 무패턴). 패턴은 patterns.parquet 의 pattern_type 이름으로 번호를 붙인다."""
    case = "CASE p.pattern_type " + " ".join(f"WHEN {q(n)} THEN {i}" for i, n in enumerate(PATTERN_NAMES[1:9], 1)) + " END"
    con.execute(f"""
        CREATE OR REPLACE TABLE pattern_cls AS
        SELECT edge_id, min({case}) AS pcls, count(DISTINCT p.pattern_type) AS n_types
        FROM read_parquet({q(paths.patterns)}) p WHERE edge_id IS NOT NULL GROUP BY edge_id""")
    bad = con.execute("SELECT count(*) FROM pattern_cls WHERE pcls IS NULL OR n_types > 1").fetchone()[0]
    assert bad == 0, f"패턴 이름이 표에 없거나 한 거래에 패턴이 둘 이상: {bad}"
    con.execute(f"""
        CREATE OR REPLACE VIEW labels AS
        SELECT t.edge_id, t.day, t.is_laundering::TINYINT AS y,
               (CASE WHEN NOT t.is_laundering THEN 0 ELSE coalesce(p.pcls, 9) END)::TINYINT AS cls
        FROM read_parquet({q(paths.tx)}) t LEFT JOIN pattern_cls p USING (edge_id)""")
    bad = con.execute("SELECT count(*) FROM labels l JOIN pattern_cls p USING (edge_id) WHERE l.y = 0").fetchone()[0]
    assert bad == 0, f"정상 라벨 거래에 패턴이 붙음: {bad}"


def val_sample(con, paths, rate=0.02, seed=0):
    """binary_classification(bc_lib.val_sample) 과 같은 검증 표본: 양성 전부(w=1) + 날짜별 정상 rate 비율(w = 그날 정상 수 / 뽑은 수).
    그쪽이 만든 파일(parquets/<DATASET>/sampling/val/...)이 있으면 그대로 읽고, 없으면 같은 규칙으로 만들어 v10_lgbm/ 에 둔다.
    돌려주는 것: edge_id, w"""
    a, b = SPLITS["val"]
    name = f"{a}_{b}_neg{rate:g}.parquet"
    for path in (paths.base / "sampling" / "val" / name, paths.out / f"val_sample_{name}"):
        if path.exists():
            v = pd.read_parquet(path, columns=["edge_id", "w"])
            log(f"검증 표본 {path}  {len(v):,} 행")
            return v
    lab = con.execute(f"""SELECT edge_id, day, is_laundering::INT AS y FROM read_parquet({q(paths.tx)})
                          WHERE day BETWEEN DATE {q(a)} AND DATE {q(b)} ORDER BY edge_id""").df()
    rng = np.random.default_rng(seed)
    parts = [lab[lab.y == 1].assign(w=1.0)]
    for day, g in lab[lab.y == 0].groupby("day"):
        k = max(1, int(round(len(g) * rate))); parts.append(g.iloc[rng.choice(len(g), k, replace=False)].assign(w=len(g) / k))
    v = pd.concat(parts).sort_values("edge_id").reset_index(drop=True)
    v.to_parquet(path, index=False)
    log(f"검증 표본 새로 만듦 {path}  {len(v):,} 행")
    return v[["edge_id", "w"]]


def build_split(con, paths, split, neg_rate=1.0, seed=42, keep=None, verbose=True):
    """split 구간의 거래를 하루씩: 엣지 피처 + 라벨 + 그날 노드 스냅샷(src·dst) → lgbm/<split>.parquet.
    neg_rate < 1 이면 정상 거래를 날짜별 고정 난수로 그 비율만 남기고 가중치 w = 1/neg_rate (양성은 전부, w=1).
    keep(edge_id, w) 를 주면 그 거래만 그 가중치로 남긴다 (val_sample 용)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    paths.lgbm.mkdir(parents=True, exist_ok=True)
    a, b = SPLITS[split]
    out = paths.lgbm / f"{split}.parquet"; tmp = out.with_name(f".{split}.partial.parquet")
    edge_cols = ", ".join(f'e."{c}"' for c in EDGE_FEATURES + ["payment_format_code", "in_port", "out_port"])
    # 라벨은 구간 것만 한 번 올려 둔다 (날짜마다 transactions.parquet 전체를 다시 읽지 않게)
    con.execute(f"""CREATE OR REPLACE TABLE split_labels AS
                    SELECT edge_id, y, cls FROM labels WHERE day BETWEEN DATE {q(a)} AND DATE {q(b)}""")
    writer = None; stats = dict(rows_full=0, rows_kept=0, positives=0, unmatched=0, days=0)
    try:
        for ds in days_between(a, b):
            df = con.execute(f"""
                SELECT e.edge_id, {day_index(ds)}::SMALLINT AS day, l.y, l.cls, e.src, e.dst, {edge_cols}
                FROM read_parquet({q(paths.edge)}) e JOIN split_labels l USING (edge_id)
                WHERE e.day = DATE {q(ds)} ORDER BY e.edge_id""").df()
            stats["rows_full"] += len(df)
            if not len(df):
                continue
            w = np.ones(len(df), np.float32)
            if keep is not None:
                df = df.merge(keep, on="edge_id", how="inner"); w = df.pop("w").to_numpy(np.float32)
            elif neg_rate < 1:
                rng = np.random.default_rng([seed, day_index(ds)])
                mask = (df.y.to_numpy() == 1) | (rng.random(len(df)) < neg_rate)
                df = df.loc[mask].reset_index(drop=True); w = np.where(df.y.to_numpy() == 1, 1.0, 1.0 / neg_rate).astype(np.float32)
            x = np.load(paths.nodes / f"{ds}.npy", mmap_mode="r")
            src, dst = df.pop("src").to_numpy(np.int64), df.pop("dst").to_numpy(np.int64)
            xs, xd = np.asarray(x[src]), np.asarray(x[dst])
            cols = {"edge_id": df.edge_id.to_numpy(np.int64), "day": df.day.to_numpy(np.int16),
                    "y": df.y.to_numpy(np.int8), "cls": df.cls.to_numpy(np.int8), "w": w}
            for c in EDGE_FEATURES:
                cols[c] = df[c].to_numpy(np.float32)
            cols["payment_format_code"] = df.payment_format_code.to_numpy(np.int8)
            cols["in_port"] = df.in_port.to_numpy(np.float32); cols["out_port"] = df.out_port.to_numpy(np.float32)
            for j, f in enumerate(NODE_FEATURES):
                cols[f"src_{f}"] = xs[:, j]
            for j, f in enumerate(NODE_FEATURES):
                cols[f"dst_{f}"] = xd[:, j]
            table = pa.table(cols)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            stats["rows_kept"] += len(df); stats["positives"] += int(cols["y"].sum())
            stats["unmatched"] += int((cols["cls"] == 9).sum()); stats["days"] += 1
            if verbose:
                log(f"{split} {ds}  전체 {stats['rows_full']:,}  저장 {stats['rows_kept']:,}  양성 {stats['positives']:,}")
            del df, x, xs, xd, cols, table
        writer.close(); writer = None
        tmp.replace(out)
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
    con.execute("DROP TABLE split_labels")
    stats.update(split=split, window=[a, b], neg_rate=neg_rate, seed=seed, file=str(out))
    return stats
