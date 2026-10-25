"""05 다운샘플링 실험 — 공통 모듈.

01/02 출력(parquets/<DATASET>/transactions.parquet, features*/edge_features.parquet, node_features.parquet)만 읽는다.
그래프 구성은 money_laundry 추론서버 binary_v10/preprocess.py 의 GraphSnapshot 과 같은 규칙:
  - 계좌(owner)마다 incoming(dst=owner) / outgoing(src=owner) 간선을 hash 순으로 최대 NEIGHBORS 개 → adjacency
  - target 의 src,dst 를 root 로 1홉 → 그 이웃으로 2홉, 간선 합집합, ego 플래그
모델은 binary_v10/network.py 의 BinaryGNN 과 같은 층 구성(ego=True, ports=False).

실험 산출물 위치: parquets/<DATASET>/sampling/
  cache/<features>[_<exclude>]/   GraphStore memmap (한 번 만들면 재사용)
  select/<method>_r<ratio>/       05_sampling_select.py 출력 (selected.parquet, meta.json)
  val/<window>_neg<rate>.parquet  고정 검증 표본 (모든 조건이 공유)
  runs/<run_name>/                05_sampling_train.py 출력 (config.json, run.json, val_scores.parquet)
"""
from __future__ import annotations
from pathlib import Path
import json, time, hashlib
import numpy as np
import pandas as pd
import duckdb

DATASET = "HI-Large"
NEIGHBORS = 100          # v10 Options.neighbors

# 시간 분할 (월~일 주 단위, 11월 이후 미사용 — 11/6 부터 생성기 꼬리 구간)
TRAIN = "2022-08-08:2022-10-02"   # 8주
VAL   = "2022-10-03:2022-10-16"   # 2주  (에폭·임계값 선택)
TEST  = "2022-10-17:2022-10-30"   # 2주  (최종 후보만 평가)
# 구간별 그래프·노드 피처 폴더 — 02 를 TAG=train/val/test, CONTEXT_END=각 구간 끝 날짜로 돌린 출력.
# 노드 피처와 adjacency 는 그 구간 끝까지의 거래로만 만든다 (미래 거래 누수 방지).
FEAT = {"train": "features_train", "val": "features_val", "test": "features_test"}
HASH_SEED = 42           # v10 adjacency hash seed

NODE_FEATURES = ["log_out_count", "log_in_count", "log_out_neighbors", "log_in_neighbors", "log_self_count",
                 "log_reciprocal_neighbors", "mean_paid_currency_z", "paid_outlier_fraction", "log_currency_count", "log_active_days",
                 "type_Corporation", "type_Partnership", "type_Sole Proprietorship", "type_Individual", "type_Country", "type_Direct",
                 "log_entity_observed_accounts", "log_entity_out_diversity", "log_entity_in_diversity"]
EDGE_FEATURES = ["pair_prior_count", "pair_gap_minutes", "pair_count_1h", "sender_gap_minutes", "sender_count_1h",
                 "receiver_gap_minutes", "receiver_count_1h", "paid_currency_z", "pair_history_count", "reverse_history_count",
                 "pair_age_at_day_start", "same_entity", "same_known_country", "both_countries_known", "same_currency",
                 "is_self_transfer", "amt_z_pay_ccy"]
assert len(NODE_FEATURES) == 19 and len(EDGE_FEATURES) == 17


# ----------------------------------------------------------------------------- 경로·유틸
def find_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "datasets").is_dir() or (p / "parquets").is_dir():
            return p
    raise FileNotFoundError("datasets/ 또는 parquets/ 가 있는 저장소 루트를 못 찾음")


ROOT = find_root()
BASE = ROOT / "parquets" / DATASET
SAMPLING = BASE / "sampling"


def q(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def log(msg: str):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def connect(threads=8, memory="12GB", path: Path | None = None):
    """memory 는 DuckDB 한도. 컨테이너 cgroup 한도(27GB) 안에서 파이썬 배열·페이지 캐시와 나눠 써야 하므로 절반 이하로.
    temp_directory 를 두어 한도를 넘는 중간 결과는 디스크로 스필한다 (메모리 DB 는 기본으로 스필 못 함)."""
    con = duckdb.connect(str(path) if path else ":memory:")
    tmp = SAMPLING / "duckdb_tmp"; tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET threads={threads}"); con.execute(f"SET memory_limit='{memory}'")
    con.execute(f"SET temp_directory={q(tmp)}"); con.execute("SET preserve_insertion_order=false")
    return con


def parse_window(s: str) -> tuple[str, str]:
    """'2022-08-08:2022-09-27' → ('2022-08-08', '2022-09-27'), 양끝 포함."""
    a, b = s.split(":")
    assert a <= b, s
    return a, b


def json_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def features_dir(name: str = "features") -> Path:
    d = BASE / name
    assert (d / "edge_features.parquet").is_file(), f"{d} 에 02 출력이 없음"
    return d


def features_context_end(name: str) -> str | None:
    """02 가 기록한 features.json 의 context_end (그 폴더 피처가 어느 날짜까지의 거래로 계산됐나)."""
    f = features_dir(name) / "features.json"
    return str(json.loads(f.read_text(encoding="utf-8")).get("context_end"))[:10] if f.is_file() else None


def check_features(name: str, window: str):
    """피처 폴더의 기준 시점이 구간 끝과 같은지 확인. 다르면 미래 거래가 섞이거나(누수) 구간 거래가 빠진다."""
    end = parse_window(window)[1]; ce = features_context_end(name)
    assert ce == end, (f"{name} 의 context_end={ce} 가 구간 끝 {end} 와 다름 — "
                       f"02 를 TAG 와 CONTEXT_END='{end}' 로 다시 돌려야 함")


# ----------------------------------------------------------------------------- 라벨
def load_labels(con, window: tuple[str, str], patterns=True) -> pd.DataFrame:
    """해당 기간 거래의 edge_id, day, src, dst, y(0/1), in_pattern(패턴 파일 매칭 여부)."""
    a, b = window
    pat = f"LEFT JOIN (SELECT DISTINCT edge_id FROM read_parquet({q(BASE/'patterns.parquet')})) p USING (edge_id)" if patterns else ""
    return con.execute(f"""
        SELECT t.edge_id, t.day, t.src, t.dst, t.is_laundering::INT AS y, {'(p.edge_id IS NOT NULL)' if patterns else 'FALSE'} AS in_pattern
        FROM read_parquet({q(BASE/'transactions.parquet')}) t {pat}
        WHERE t.day BETWEEN DATE {q(a)} AND DATE {q(b)}
        ORDER BY t.edge_id""").df()


def read_edge_features(con, feat_dir: Path, edge_ids: np.ndarray, extra_cols=()) -> pd.DataFrame:
    """edge_features.parquet 에서 주어진 edge_id 행만 edge_id 순으로 (17피처 + extra_cols).
    memmap 무작위 접근은 페이지 캐시를 수 GB 먹어 cgroup 한도를 넘기므로, 큰 집합은 이걸로 읽는다 (parquet 스캔 1회)."""
    con.register("want_ids", pd.DataFrame({"edge_id": np.asarray(edge_ids, np.int64)}))
    cols = ", ".join(f'e."{c}"' for c in list(EDGE_FEATURES) + list(extra_cols))
    df = con.execute(f"SELECT e.edge_id, {cols} FROM read_parquet({q(feat_dir / 'edge_features.parquet')}) e JOIN want_ids USING (edge_id) ORDER BY e.edge_id").df()
    con.unregister("want_ids")
    assert len(df) == len(np.unique(edge_ids)), f"피처 누락 {len(np.unique(edge_ids)) - len(df)}건"
    return df


# ----------------------------------------------------------------------------- 그래프 저장소
class GraphStore:
    """v10 GraphSnapshot 과 같은 2홉 그래프를 numpy memmap 으로 만든다.

    prepare(): DuckDB 로 memmap 캐시 생성 (edge_x, edge_src, edge_dst, node_x, adjacency). 1회.
    load():    memmap 열기. build(edge_id) / collate(edge_ids) 로 배치 생성.

    exclude_parquet: 구조 샘플링(apply=struct)용. 여기 있는 edge_id 는 adjacency 에서 제외된다
                     (= 메시지 패싱 그래프에서 삭제). 피처는 그대로(구조-A). 피처까지 다시 계산하려면
                     02 를 다른 TAG 로 돌리고 features_name 을 그 폴더로 준다(구조-B).
    """

    def __init__(self, features_name="features", exclude_parquet: Path | None = None, threads=8, memory="12GB"):
        self.feat = features_dir(features_name)
        self.context_end = features_context_end(features_name)     # adjacency 도 이 날짜까지의 거래로만
        self.exclude = Path(exclude_parquet) if exclude_parquet else None
        tag = features_name + ("_" + hashlib.sha1(str(self.exclude).encode()).hexdigest()[:8] if self.exclude else "")
        self.cache = SAMPLING / "cache" / tag
        self.threads, self.memory = threads, memory

    # --- 준비 -------------------------------------------------------------------
    def prepare(self, force=False):
        self.cache.mkdir(parents=True, exist_ok=True)
        done = self.cache / "ready.json"
        if done.exists() and not force:
            log(f"GraphStore 캐시 있음: {self.cache}"); return
        con = connect(self.threads, self.memory)
        tx = BASE / "transactions.parquet"
        n_edge = con.execute(f"SELECT max(edge_id) FROM read_parquet({q(self.feat/'edge_features.parquet')})").fetchone()[0] + 1
        n_node = con.execute(f"SELECT count(*) FROM read_parquet({q(self.feat/'node_features.parquet')})").fetchone()[0]
        log(f"memmap 생성: edge {n_edge:,} × 17, node {n_node:,} × 19, 기준 시점 {self.context_end} → {self.cache}")

        # 엣지 피처 / 끝점 (edge_id 로 직접 인덱싱, 청크로 채움)
        ex = np.lib.format.open_memmap(self.cache / "edge_x.npy", "w+", np.float32, (n_edge, 17))
        es = np.lib.format.open_memmap(self.cache / "edge_src.npy", "w+", np.int32, (n_edge,))
        ed = np.lib.format.open_memmap(self.cache / "edge_dst.npy", "w+", np.int32, (n_edge,))
        cols = ", ".join(f'"{c}"' for c in EDGE_FEATURES)
        step = 10_000_000
        for a in range(0, n_edge, step):
            df = con.execute(f"""SELECT edge_id, src, dst, {cols} FROM read_parquet({q(self.feat/'edge_features.parquet')})
                                 WHERE edge_id >= {a} AND edge_id < {a+step} ORDER BY edge_id""").df()
            if len(df):
                i = df.edge_id.to_numpy()
                ex[i] = df[EDGE_FEATURES].to_numpy(np.float32); es[i] = df.src.to_numpy(np.int32); ed[i] = df.dst.to_numpy(np.int32)
            log(f"  edge {min(a+step, n_edge):,}/{n_edge:,}")
        ex.flush(); es.flush(); ed.flush()

        # 노드 피처 (node_id = 0..N-1 로 정렬돼 있음)
        ncols = ", ".join(f'"{c}"' for c in NODE_FEATURES)
        nx = con.execute(f"SELECT node_id, {ncols} FROM read_parquet({q(self.feat/'node_features.parquet')}) ORDER BY node_id").df()
        assert np.array_equal(nx.node_id.to_numpy(), np.arange(n_node)), "node_id 가 0..N-1 이 아님"
        np.save(self.cache / "node_x.npy", nx[NODE_FEATURES].to_numpy(np.float32))

        # adjacency: v10 과 같은 규칙 (owner·방향별 hash 순 상위 NEIGHBORS)
        excl = f"ANTI JOIN (SELECT edge_id FROM read_parquet({q(self.exclude)})) x USING (edge_id)" if self.exclude else ""
        adj = con.execute(f"""
            WITH t AS (SELECT edge_id, src, dst FROM read_parquet({q(tx)}) {excl}
                       {f"WHERE day <= DATE {q(self.context_end)}" if self.context_end else ""}),
                 e AS (SELECT edge_id, dst AS owner, TRUE AS incoming FROM t
                       UNION ALL SELECT edge_id, src AS owner, FALSE FROM t)
            SELECT owner, edge_id, incoming FROM e
            QUALIFY row_number() OVER (PARTITION BY owner, incoming ORDER BY hash(edge_id, owner, {HASH_SEED}), edge_id) <= {NEIGHBORS}
            ORDER BY owner, edge_id""").df()
        owners = adj.owner.to_numpy(np.int64)
        ptr = np.zeros(n_node + 1, np.int64)
        np.add.at(ptr, owners + 1, 1); ptr = np.cumsum(ptr)
        np.save(self.cache / "adj_ptr.npy", ptr)
        np.save(self.cache / "adj_edge.npy", adj.edge_id.to_numpy(np.int32))
        np.save(self.cache / "adj_in.npy", adj.incoming.to_numpy(bool))
        json_write(done, dict(features=str(self.feat), context_end=self.context_end, exclude=str(self.exclude), n_edge=n_edge, n_node=n_node,
                              adjacency_rows=len(adj), neighbors=NEIGHBORS, hash_seed=HASH_SEED))
        con.close(); log("GraphStore 준비 끝")

    # --- 사용 -------------------------------------------------------------------
    def load(self):
        c = self.cache
        self.edge_x = np.load(c / "edge_x.npy", mmap_mode="r")
        self.edge_src = np.load(c / "edge_src.npy", mmap_mode="r")
        self.edge_dst = np.load(c / "edge_dst.npy", mmap_mode="r")
        self.node_x = np.load(c / "node_x.npy")                 # 2.1M × 19 — 메모리에 올려도 됨
        self.ptr = np.load(c / "adj_ptr.npy")
        self.adj_edge = np.load(c / "adj_edge.npy", mmap_mode="r")
        self.adj_in = np.load(c / "adj_in.npy", mmap_mode="r")
        return self

    def _select(self, nodes: np.ndarray):
        """owner 집합의 adjacency 행 → (edge_ids, incoming, outgoing)  (v10 _select 와 같음)"""
        rows = np.concatenate([np.arange(self.ptr[n], self.ptr[n + 1]) for n in np.unique(nodes)] or [np.zeros(0, np.int64)])
        e = np.asarray(self.adj_edge[rows], np.int64); inc = np.asarray(self.adj_in[rows])
        edge_ids, inv = np.unique(e, return_inverse=True)
        incoming = np.zeros(len(edge_ids), bool); outgoing = np.zeros(len(edge_ids), bool)
        np.logical_or.at(incoming, inv, inc); np.logical_or.at(outgoing, inv, ~inc)
        return edge_ids, incoming, outgoing

    def build(self, edge_id: int) -> dict:
        """target 1건의 2홉 서브그래프 (v10 GraphSnapshot.build 와 같은 규칙)."""
        src, dst = int(self.edge_src[edge_id]), int(self.edge_dst[edge_id])
        roots = np.unique([src, dst])
        e1, in1, out1 = self._select(roots)
        s1, d1 = np.asarray(self.edge_src[e1], np.int64), np.asarray(self.edge_dst[e1], np.int64)
        neighbors = np.unique(np.r_[s1[in1], d1[out1]])
        e2, in2, out2 = self._select(neighbors)
        e_all, inv = np.unique(np.r_[e1, e2], return_inverse=True)
        incoming = np.zeros(len(e_all), bool); outgoing = np.zeros(len(e_all), bool)
        np.logical_or.at(incoming, inv, np.r_[in1, in2]); np.logical_or.at(outgoing, inv, np.r_[out1, out2])
        s, d = np.asarray(self.edge_src[e_all], np.int64), np.asarray(self.edge_dst[e_all], np.int64)
        nodes = np.unique(np.r_[roots, s, d])
        ego = np.isin(nodes, roots).astype(np.float32)
        return dict(
            x=np.column_stack([self.node_x[nodes], ego]),                      # [n, 20]
            edge_index=np.stack([np.searchsorted(nodes, s), np.searchsorted(nodes, d)]),  # [2, m]
            edge_attr=np.asarray(self.edge_x[e_all], np.float32),              # [m, 17]
            target_index=np.array([np.searchsorted(nodes, src), np.searchsorted(nodes, dst)]),
            target_attr=np.asarray(self.edge_x[edge_id], np.float32),           # [17]
            incoming=incoming, outgoing=outgoing)

    def collate(self, edge_ids, device="cpu"):
        """여러 target 서브그래프를 하나의 분리 그래프로 합쳐 BinaryGNN 입력 텐서로."""
        import torch
        xs, eis, eas, tis, tas, ins, outs, off = [], [], [], [], [], [], [], 0
        for e in edge_ids:
            g = self.build(int(e))
            xs.append(g["x"]); eis.append(g["edge_index"] + off); eas.append(g["edge_attr"])
            tis.append(g["target_index"] + off); tas.append(g["target_attr"]); ins.append(g["incoming"]); outs.append(g["outgoing"])
            off += len(g["x"])
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
        return (t(np.concatenate(xs), torch.float32), t(np.concatenate(eis, 1), torch.long), t(np.concatenate(eas), torch.float32),
                t(np.stack(tis, 1), torch.long), t(np.stack(tas), torch.float32),
                t(np.concatenate(ins), torch.bool), t(np.concatenate(outs), torch.bool))


# ----------------------------------------------------------------------------- 모델 (binary_v10/network.py 와 같은 층 구성)
def make_model(pos_weight: float):
    import torch
    from torch_geometric.nn import TransformerConv

    class BinaryGNN(torch.nn.Module):
        def __init__(self, node_dim=20, edge_dim=17):
            super().__init__()
            self.conv1 = TransformerConv(node_dim, 64, heads=2, edge_dim=edge_dim, dropout=0.0)
            self.conv2 = TransformerConv(128, 64, heads=2, edge_dim=edge_dim, dropout=0.0)
            self.reverse1 = TransformerConv(node_dim, 64, heads=2, edge_dim=edge_dim, dropout=0.0, root_weight=False)
            self.reverse2 = TransformerConv(128, 64, heads=2, edge_dim=edge_dim, dropout=0.0, root_weight=False)
            self.merge1 = torch.nn.Linear(256, 128, bias=False)
            self.merge2 = torch.nn.Linear(256, 128, bias=False)
            self.head = torch.nn.Sequential(torch.nn.Linear(256 + edge_dim, 128), torch.nn.ReLU())
            self.risk_head = torch.nn.Linear(128, 1)
            # v10: 점수는 sigmoid(logit - ln(pos_weight)) — pos_weight 로 밀린 확률을 되돌림
            self.register_buffer("risk_offset", torch.tensor(-np.log(pos_weight), dtype=torch.float32))

        def forward(self, x, edge_index, edge_attr, target_index, target_attr, incoming, outgoing):
            direct, direct_attr = edge_index[:, incoming], edge_attr[incoming]
            reverse, reverse_attr = edge_index[:, outgoing].flip(0), edge_attr[outgoing]
            h = self.merge1(torch.cat([self.conv1(x, direct, direct_attr), self.reverse1(x, reverse, reverse_attr)], 1)).relu()
            z = self.merge2(torch.cat([self.conv2(h, direct, direct_attr), self.reverse2(h, reverse, reverse_attr)], 1)).relu()
            z = self.head(torch.cat([z[target_index[0]], z[target_index[1]], target_attr], 1))
            return self.risk_head(z).squeeze(-1)

        def scores(self, *batch):
            return (self(*batch) + self.risk_offset).sigmoid()

    return BinaryGNN()


# ----------------------------------------------------------------------------- 지표 (검증 표본 가중치 w 반영)
def evaluate(y: np.ndarray, score: np.ndarray, w: np.ndarray, unmatched: np.ndarray, budget=0.01, bins=15) -> dict:
    from sklearn.metrics import average_precision_score, precision_recall_curve
    y = y.astype(int); out = {"AP": float(average_precision_score(y, score, sample_weight=w))}
    p, r, th = precision_recall_curve(y, score, sample_weight=w)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    k = int(np.argmax(f1)); out.update(F1=float(f1[k]), precision=float(p[k]), recall=float(r[k]), threshold=float(th[k]))
    o = np.argsort(-score); cw = np.cumsum(w[o]); cut = cw <= budget * w.sum()          # 상위 budget 비율 알림 예산
    out["recall_at_1pct"] = float((w[o] * y[o])[cut].sum() / max((w * y).sum(), 1e-12))
    um = unmatched.astype(bool)
    out["unmatched_recall_at_1pct"] = float((w[o] * (y[o] & um[o]))[cut].sum() / max((w * (y & um)).sum(), 1e-12))
    edges = np.linspace(0, 1, bins + 1); b = np.clip(np.digitize(score, edges) - 1, 0, bins - 1); ece = 0.0
    for i in range(bins):
        m = b == i
        if w[m].sum() > 0:
            ece += w[m].sum() / w.sum() * abs(np.average(score[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    out["ECE"] = float(ece)
    return out
