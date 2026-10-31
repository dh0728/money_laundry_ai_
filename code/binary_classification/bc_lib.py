"""binary_classification 공통 모듈 — 01_basic_features_baseline.ipynb 에서 검증된 코드를 그대로 함수로 옮긴 것.

02 부터의 노트북은 "피처 계산 셀" 만 자기 것이고 나머지(데이터 로드, port, 그래프, GINe, 학습, 테스트, 분석)는 여기 함수를 부른다.
01 노트북 자체는 건드리지 않는다 (원본 기록).

로직은 01 과 같다:
  - 기본 피처 4개 + port 2개 (Multi-GNN), z-norm 은 학습 구간 통계
  - hetero 그래프 to / rev_to, 도착 노드순 정렬 + is_sorted=True, 그래프는 한 번에 하나만 (use_graph)
  - GINe (to_hetero aggr='mean' 과 같은 꼴), 가중 CE, 배치 256개 = 1 에폭, EVAL_PERIOD 마다 검증, 검증 AP 로 선택
  - 검증은 05 와 같은 가중 표본, 테스트는 구간 전체

분석 (0단계):
  - pattern_recall   : 패턴 유형별 recall@1% (patterns.parquet 조인)
  - degree_stratified: 수신 계좌 in-degree 구간별 F1 / AP (SALT-GNN 프로토콜)
"""
from __future__ import annotations
import sys, json, time, gc, ctypes
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

# 05 샘플링 실험 공통 모듈 (경로, 분할, 지표)
sys.path.append(str(Path(__file__).resolve().parent.parent))
import sampling_lib as L

TO, REV = ('node', 'to', 'node'), ('node', 'rev_to', 'node')
BASIC_EDGE_FEATURES = ['timestamp', 'amount_received', 'received_currency', 'payment_format']
PORT_FEATURES = ['in_port', 'out_port']
VAL_KEYS = ('AP', 'F1', 'F1_at_0.5', 'precision', 'recall', 'threshold', 'recall_at_1pct', 'unmatched_recall_at_1pct', 'ECE')
WINDOWS = {
    'HI-Large': dict(train=L.TRAIN, val=L.VAL, test=L.TEST),
    'HI-Small': dict(train='2022-09-01:2022-09-06', val='2022-09-07:2022-09-08', test='2022-09-09:2022-09-10'),
}


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def trim(label=''):
    """지운 메모리를 OS 에 돌려주고(glibc 는 안 돌려줄 때가 많다) 현재 RSS 를 찍는다.
    pyarrow 풀(mimalloc/jemalloc)도 parquet 을 읽고 난 메모리를 쥐고 있어서 같이 돌려준다"""
    gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)
    try:
        import pyarrow as pa; pa.default_memory_pool().release_unused()
    except Exception:
        pass
    print(f'RSS {psutil.Process().memory_info().rss / 1e9:.1f} GB {label}')


def release_mmap(arr):
    """memmap 이 읽어 들인 파일 페이지를 이 프로세스의 상주 집합에서 뺀다 (MADV_DONTNEED). 데이터는 디스크/페이지 캐시에 그대로.
    6.5GB 행렬을 한 번 훑으면 RSS 가 그만큼 올라 보이는데, 깨끗한 파일 페이지라 OOM kill 대상은 아니지만 숫자를 정직하게 보려고"""
    import mmap
    try:
        if isinstance(arr, np.memmap) and arr._mmap is not None:
            arr._mmap.madvise(mmap.MADV_DONTNEED)
    except Exception:
        pass


# ============================================================================ 설정
@dataclass
class Config:
    dataset: str = 'HI-Large'
    smoke: bool = False
    val_neg_rate: float = 0.02
    # 트리
    scale_pos_weight: float = 6
    tree_rounds: int = 500
    tree_early_stop: int = 20
    # GNN (Multi-GNN model_settings 'gin')
    num_neighbors: list = field(default_factory=lambda: [100, 100])
    batch_size: int = 2048
    eval_batch_size: int = 8192       # 검증·테스트 추론 배치 (역전파 없음 → 크게. 결과 동일, 배치 수 1/4)
    iter_per_epoch: int = 256
    epochs: int = 400
    eval_period: int = 4
    patience: int = 8
    class_weight: list = field(default_factory=lambda: [1.0, 6.27])
    n_hidden: int = 66
    n_gnn_layers: int = 2
    lr: float = 0.006
    dropout: float = 0.01
    final_dropout: float = 0.1
    num_workers: int = 0
    seed: int = 42

    def __post_init__(self):
        w = WINDOWS[self.dataset]
        self.train_window, self.val_window, self.test_window = w['train'], w['val'], w['test']
        if self.smoke:
            self.iter_per_epoch, self.epochs, self.tree_rounds, self.eval_period = 8, 2, 20, 1
        # sampling_lib 은 DATASET 을 모듈 로드 시점에 읽으므로 경로를 맞춰 준다
        L.DATASET = self.dataset; L.BASE = L.ROOT / 'parquets' / self.dataset; L.SAMPLING = L.BASE / 'sampling'
        self.tx_path = L.BASE / 'transactions.parquet'
        self.feat_dir = L.BASE / 'features_bc'          # 단계별 피처 parquet (edge_id 순)
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        assert self.tx_path.is_file(), f'01_prepare_parquet 을 {self.dataset} 으로 먼저 실행: {self.tx_path}'

    def out_dir(self, name):
        d = Path.cwd() / 'results' / self.dataset / name; d.mkdir(parents=True, exist_ok=True); return d


def seed_all(seed):
    import random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)


# ============================================================================ 1. 데이터
def code_tables(con, cfg):
    """통화·결제 방식 → 정수 코드. 학습 구간에서 본 값에 0,1,2,... 못 본 값은 -1"""
    a, b = L.parse_window(cfg.train_window)
    def codes(col):
        return con.execute(f"""
            SELECT {col} AS v, row_number() OVER (ORDER BY {col}) - 1 AS code
            FROM (SELECT DISTINCT {col} FROM read_parquet({L.q(cfg.tx_path)}) WHERE day BETWEEN DATE {L.q(a)} AND DATE {L.q(b)})""").df()
    currency_codes, format_codes = codes('receiving_currency'), codes('payment_format')
    con.register('currency_codes', currency_codes); con.register('format_codes', format_codes)
    return currency_codes, format_codes


def load_tx_lowmem(con, cfg, port_path=None):
    """load_tx 의 저메모리 판 (03 부터). DuckDB 로 한 번에 .df() 하면 DuckDB 결과(최대 12GB) + pandas 가 동시에 살아 피크가 20GB 를 넘는다 (03 에서 RSS 25.5GB).
    대신 ① 기본 열을 edge_id 순 parquet(features_bc/tx_base.parquet) 으로 한 번 COPY 해 두고 (스트리밍이라 DuckDB 메모리 작음)
         ② 열 하나씩 numpy 로 읽어 DataFrame 을 만든다 (피크 = 완성된 tx + 열 하나).  timestamp 는 float32 (초 단위 7.8e6 까지 정확)
    port_path 를 주면 port 두 열도 int32 로 같이 붙인다. 코드표(currency_codes, format_codes)는 con 에 등록돼 있어야 한다"""
    import pyarrow.parquet as pq
    base = cfg.feat_dir / 'tx_base.parquet'
    if not base.exists():
        t0 = time.time()
        con.execute(f"""
            COPY (
                SELECT t.edge_id, t.src::INT AS src, t.dst::INT AS dst, CAST(t.day AS TIMESTAMP) AS day,
                       (epoch(t.ts) - (SELECT epoch(min(ts)) FROM read_parquet({L.q(cfg.tx_path)})))::FLOAT AS timestamp,
                       t.received::FLOAT AS amount_received,
                       coalesce(c.code, -1)::SMALLINT AS received_currency,
                       coalesce(f.code, -1)::SMALLINT AS payment_format,
                       t.is_laundering::TINYINT AS y
                FROM read_parquet({L.q(cfg.tx_path)}) t
                LEFT JOIN currency_codes c ON c.v = t.receiving_currency
                LEFT JOIN format_codes   f ON f.v = t.payment_format
                ORDER BY t.edge_id
            ) TO {L.q(base)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
        log(f'tx_base.parquet 생성 {time.time()-t0:.0f}s')
    cols = {}
    for c in ['edge_id', 'src', 'dst', 'day', 'timestamp', 'amount_received', 'received_currency', 'payment_format', 'y']:
        arr = pq.read_table(base, columns=[c]).column(c).to_numpy()
        cols[c] = arr.astype('datetime64[ns]') if c == 'day' else arr
        del arr
    tx = pd.DataFrame(cols, copy=False); del cols; gc.collect()
    if port_path is not None:
        tx = attach_parquet(tx, port_path, PORT_FEATURES, dtype=np.int32)
    return tx


def load_tx(con, cfg):
    """거래 전체 (edge_id 순): edge_id, src, dst, day, 기본 피처 4, y.  문자열 열은 pandas 에 올리지 않는다 (코드 테이블 조인)
    (01·02 가 쓰는 판. 피처가 많아지는 03 부터는 load_tx_lowmem)"""
    return con.execute(f"""
        SELECT t.edge_id, t.src::INT AS src, t.dst::INT AS dst, CAST(t.day AS TIMESTAMP) AS day,
               epoch(t.ts) - (SELECT epoch(min(ts)) FROM read_parquet({L.q(cfg.tx_path)}))  AS timestamp,
               t.received::FLOAT                        AS amount_received,
               coalesce(c.code, -1)::SMALLINT           AS received_currency,
               coalesce(f.code, -1)::SMALLINT           AS payment_format,
               t.is_laundering::TINYINT                 AS y
        FROM read_parquet({L.q(cfg.tx_path)}) t
        LEFT JOIN currency_codes c ON c.v = t.receiving_currency
        LEFT JOIN format_codes   f ON f.v = t.payment_format
        ORDER BY t.edge_id""").df()


def port_numbers(owner, other, t, n_nodes):
    """owner 계좌 안에서 상대(other)를 첫 거래 시각순으로 번호 매긴 값을 간선마다 돌려준다 (Multi-GNN ports())"""
    key = owner.astype(np.int64) * n_nodes + other
    order = np.lexsort((t, key)); key_s = key[order]
    first = np.empty(len(key_s), bool); first[0] = True; np.not_equal(key_s[1:], key_s[:-1], out=first[1:])
    pair_key = key_s[first]; pair_t = t[order][first]
    del order, key_s, first
    p_owner, p_other = pair_key // n_nodes, pair_key % n_nodes
    o2 = np.lexsort((p_other, pair_t, p_owner)); po = p_owner[o2]
    start = np.empty(len(po), bool); start[0] = True; np.not_equal(po[1:], po[:-1], out=start[1:])
    grp_start = np.maximum.accumulate(np.where(start, np.arange(len(po)), 0))
    rank = np.empty(len(po), np.int32); rank[o2] = (np.arange(len(po)) - grp_start).astype(np.int32)
    del o2, po, start, grp_start, p_owner, p_other, pair_t
    return rank[np.searchsorted(pair_key, key)]


def ensure_ports(con, cfg):
    """features_ports/ports.parquet 이 없으면 만든다 (tx 로드 전에 호출: numpy 로 최대 약 11GB)"""
    path = L.BASE / 'features_ports' / 'ports.parquet'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
        arr = con.execute(f"SELECT edge_id, src::INT AS src, dst::INT AS dst, epoch(ts)::BIGINT AS t FROM read_parquet({L.q(cfg.tx_path)}) ORDER BY edge_id").fetchnumpy()
        n_nodes = int(max(arr['src'].max(), arr['dst'].max())) + 1
        df = pd.DataFrame(dict(edge_id=arr['edge_id'], in_port=port_numbers(arr['dst'], arr['src'], arr['t'], n_nodes),
                               out_port=port_numbers(arr['src'], arr['dst'], arr['t'], n_nodes)))
        del arr; gc.collect(); df.to_parquet(path, index=False); del df; gc.collect()
        log(f'ports.parquet 생성 {time.time()-t0:.0f}s')
    return path


def attach_parquet(tx, path, cols, dtype=None):
    """edge_id 순으로 저장된 피처 parquet 의 열들을 tx 에 붙인다 (행 순서가 같은지 확인). DuckDB 조인은 1.8억 × 1.8억이라 메모리 초과.
    dtype=np.float16 이면 붙이면서 반으로 줄인다 (ln 변환된 피처라 충분. 1.8억 행 × 열당 0.36GB). 열을 하나씩 읽어 피크를 낮춘다"""
    import pyarrow.parquet as pq
    ids = pq.read_table(path, columns=['edge_id']).column('edge_id').to_numpy()
    assert (ids == tx['edge_id'].to_numpy()).all(), f'{path.name} 행 순서가 tx 와 다름'
    del ids
    for c in cols:
        arr = pq.read_table(path, columns=[c]).column(c).to_numpy()
        tx[c] = arr.astype(dtype) if dtype is not None else arr
        del arr
    gc.collect()
    return tx


def window_mask(df, window):
    a, b = L.parse_window(window)
    return ((df['day'] >= pd.Timestamp(a)) & (df['day'] <= pd.Timestamp(b))).to_numpy()


def znorm_features(tx, cols, in_train, out_dtype=np.float16, out_path=None):
    """tx 의 cols 를 열 하나씩 float32 로 뽑아 학습 구간 통계로 z-norm 한 뒤 out_dtype(기본 float16) 행렬에 넣는다. 뽑은 열은 그때그때 tx 에서 떨어뜨린다.
    (01·02 는 float32 행렬이었다 — 18열이면 13GB 라 03 부터 float16. GraphStore 의 FEAT16 과 같은 dtype 이라 복사도 없다)
    (tx[cols].to_numpy() 는 dtype 이 섞여 float64 로 먼저 합쳐 1.8억 × 6 이면 8.6GB → 열 단위로)"""
    if out_path is not None:   # 디스크(.npy) 에 바로 쓰는 memmap → 행렬이 RSS 를 차지하지 않는다 (트리·GraphStore 가 그대로 읽는다)
        feat_z = np.lib.format.open_memmap(Path(out_path), mode='w+', dtype=out_dtype, shape=(len(tx), len(cols)))
    else:
        feat_z = np.empty((len(tx), len(cols)), out_dtype)
    mu = np.empty(len(cols), np.float32); sd = np.empty(len(cols), np.float32)
    for j, col in enumerate(cols):
        v = tx[col].to_numpy().astype(np.float32)
        mu[j] = v[in_train].mean(); sd[j] = v[in_train].std() + 1e-6
        v -= mu[j]; v /= sd[j]
        feat_z[:, j] = v
        del v; tx.drop(columns=[col], inplace=True)
    if out_path is not None:
        feat_z.flush(); feat_z = np.load(Path(out_path), mmap_mode='r')   # 더티 페이지를 내려 쓰고 읽기 전용으로 다시 연다
    gc.collect()
    return feat_z, pd.DataFrame({'feature': cols, 'mean(train)': mu, 'std(train)': sd})


# ============================================================================ 2. 검증·테스트 표본
def val_sample(con, window, rate, seed=0):
    """고정 검증 표본 (05_sampling_train.val_sample 과 같은 규칙·같은 파일). 없으면 만든다."""
    a, b = L.parse_window(window)
    path = L.SAMPLING / 'val' / f'{a}_{b}_neg{rate:g}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    lab = L.load_labels(con, (a, b)); rng = np.random.default_rng(seed)
    parts = [lab[lab.y == 1].assign(w=1.0)]
    for day, g in lab[lab.y == 0].groupby('day'):
        k = max(1, int(round(len(g) * rate))); parts.append(g.iloc[rng.choice(len(g), k, replace=False)].assign(w=len(g) / k))
    v = pd.concat(parts).sort_values('edge_id').reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True); v.to_parquet(path, index=False)
    return v


def test_full(con, window):
    """테스트: 구간 전체, 가중치 1 (날짜 열은 수천만 개 파이썬 객체라 뺌)"""
    return L.load_labels(con, L.parse_window(window))[['edge_id', 'dst', 'y', 'in_pattern']].assign(w=1.0)   # dst 는 degree 층화용


def rows_of(all_edge_ids, edge_ids):
    pos = np.searchsorted(all_edge_ids, edge_ids)
    assert (all_edge_ids[pos] == edge_ids).all(), 'tx 에 없는 edge_id'
    return pos


def evaluate_sample(sample_df, score):
    """05 와 같은 가중 지표 + 논문 비교용 F1@0.5 (임계값 0.5 고정, 가중)"""
    y = sample_df['y'].to_numpy(); w = sample_df['w'].to_numpy()
    unmatched = (y == 1) & ~sample_df['in_pattern'].to_numpy()
    m = L.evaluate(y, score, w, unmatched)
    pred = score >= 0.5
    tp = (w * (pred & (y == 1))).sum(); fp = (w * (pred & (y == 0))).sum(); fn = (w * (~pred & (y == 1))).sum()
    m['F1_at_0.5'] = float(2 * tp / max(2 * tp + fp + fn, 1e-12))
    return m


# ============================================================================ 3. 트리 모델
def rows_matrix(feat_z, rows, cols, dtype=np.float32, chunk=10_000_000):
    """feat_z[rows][:, cols] 를 임시 복사 없이 만든다 (행 묶음으로 읽어 바로 채움). memmap 이면 순서대로 읽어 페이지 캐시 친화적.
    (feat_z[mask][:, cols].astype(f32) 는 1억 행 × 18열에서 중간 배열 3개 = 13GB 가 한꺼번에 생겨 03 에서 RSS 27GB)"""
    rows = np.asarray(rows); X = np.empty((len(rows), len(cols)), dtype)
    for a in range(0, len(rows), chunk):
        X[a:a + chunk] = feat_z[rows[a:a + chunk]][:, cols]
    release_mmap(feat_z)
    return X


def predict_rows(model, feat_z, rows, cols, chunk=5_000_000):
    """테스트 2,700 만 행을 통째로 float32 로 만들지 않고 묶음으로 예측"""
    rows = np.asarray(rows); out = np.empty(len(rows), np.float32)
    for a in range(0, len(rows), chunk):
        out[a:a + chunk] = model.predict_proba(rows_matrix(feat_z, rows[a:a + chunk], cols))[:, 1]
    release_mmap(feat_z)
    return out


def train_trees(X_train, y_train, X_val, y_val, w_val, X_test, val_df, test_df, cfg, results, n_jobs=8):
    """X_test 는 행렬이거나, 모델을 받아 테스트 점수를 돌려주는 함수 (lambda m: predict_rows(m, feat_z, test_idx, tree_cols)) — 03 부터 후자"""
    import lightgbm as lgb, xgboost as xgb
    test_score = (lambda m: X_test(m)) if callable(X_test) else (lambda m: m.predict_proba(X_test)[:, 1])
    t0 = time.time()
    # max_delta_step: 잎 출력 상한 (최종 잎 값 ≤ lr × 0.7). 양성 0.1% + boost_from_average(초기 p≈0.006) 에서 순수 양성 잎의 잎 값이
    # −G/H ≈ 1/p ≈ 160 으로 폭주 → 트리 1개로 확률이 1.0 에 포화(검증 표본 2.7%, 그중 97% 양성)되고 AP 가 더 안 올라 best_iteration=1 로 끝났다.
    # reg_lambda=1 로는 못 막음 (G/H 비가 160 이라 λ=1 은 무의미). reg_lambda·min_child_weight 는 XGBoost 기본값과 맞춘 것
    lgb_model = lgb.LGBMClassifier(n_estimators=cfg.tree_rounds, learning_rate=0.05, num_leaves=63, scale_pos_weight=cfg.scale_pos_weight,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=1.0, reg_lambda=1.0, min_child_weight=1.0, max_delta_step=0.7,
                                   random_state=cfg.seed, n_jobs=n_jobs, verbose=-1)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_sample_weight=[w_val], eval_metric='average_precision',
                  callbacks=[lgb.early_stopping(cfg.tree_early_stop, first_metric_only=True, verbose=False)])  # AP 만 보고 멈춘다 (기본값은 logloss 도 같이 봐서 2라운드에 끊김)
    results['lightgbm'] = dict(val=evaluate_sample(val_df, lgb_model.predict_proba(X_val)[:, 1]),
                               test=evaluate_sample(test_df, test_score(lgb_model)),
                               best_iteration=int(lgb_model.best_iteration_ or cfg.tree_rounds), train_sec=round(time.time() - t0))
    log(f"lightgbm val AP {results['lightgbm']['val']['AP']:.4f}")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(n_estimators=cfg.tree_rounds, learning_rate=0.05, max_depth=8, scale_pos_weight=cfg.scale_pos_weight,
                                  subsample=0.8, colsample_bytree=1.0, tree_method='hist', device='cpu',
                                  eval_metric='aucpr', early_stopping_rounds=cfg.tree_early_stop, random_state=cfg.seed, n_jobs=n_jobs)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val], verbose=False)
    results['xgboost'] = dict(val=evaluate_sample(val_df, xgb_model.predict_proba(X_val)[:, 1]),
                              test=evaluate_sample(test_df, test_score(xgb_model)),
                              best_iteration=int(xgb_model.best_iteration), train_sec=round(time.time() - t0))
    log(f"xgboost  val AP {results['xgboost']['val']['AP']:.4f}")
    return lgb_model, xgb_model


# ============================================================================ 4. 그래프
class GraphStore:
    """그래프 재료(SRC, DST, Y, FEAT16)를 들고 학습·검증·테스트 그래프를 **한 번에 하나만** 올린다 (01 의 use_graph).
    FEAT16 열 순서: 기본 4 + in_port + out_port + (단계별 추가 피처). rev_to 간선은 port 두 열을 바꿔 붙인다.
    메모리: 그래프에는 엣지 피처를 복사해 넣지 않는다. 간선마다 tx 행 번호(`orig`, int32)만 두고 배치 변환 때 FEAT16[orig] 로 꺼낸다.
    (01 처럼 to / rev_to 에 각각 복사하면 피처 10열 기준 학습 그래프 +4.6GB → 02 에서 24GB 로 커널 사망. 모델 입력은 동일)"""

    def __init__(self, tx, feat_z, cfg, train_rows, val_idx, test_idx, port_cols, feat_memmap=None):
        """feat_memmap: 경로를 주면 피처 행렬을 .npy 로 내려 쓰고 memmap 으로 읽는다 → 1.8억 × 18열 float16 6.5GB 가 RSS 에서 빠진다
        (페이지 캐시는 cgroup 이 압박 시 회수하므로 OOM kill 대상이 아님). 배치마다 F[orig] 를 디스크/캐시에서 모아 읽는다"""
        import torch
        self.cfg = cfg
        self.NUM_NODES = int(max(tx['src'].max(), tx['dst'].max())) + 1
        self.N = {k: self._rows_until(tx, w) for k, w in (('train', cfg.train_window), ('val', cfg.val_window), ('test', cfg.test_window))}
        self.SRC = tx['src'].to_numpy().copy(); self.DST = tx['dst'].to_numpy().copy(); self.Y = tx['y'].to_numpy().astype(np.uint8)
        if isinstance(feat_z, np.memmap) and feat_z.dtype == np.float16:
            self.FEAT16 = feat_z                                        # znorm_features(out_path=…) 가 만든 memmap 그대로
            log(f'피처 행렬 memmap(기존) {tuple(self.FEAT16.shape)}')
        elif feat_memmap is not None:
            feat_memmap = Path(feat_memmap)
            if not (feat_memmap.exists() and np.load(feat_memmap, mmap_mode='r').shape == feat_z.shape):
                np.save(feat_memmap, np.asarray(feat_z, dtype=np.float16))
            self.FEAT16 = np.load(feat_memmap, mmap_mode='r')          # numpy memmap (float16). 변환에서 numpy 인덱싱
            log(f'피처 행렬 memmap {feat_memmap.name} {tuple(self.FEAT16.shape)}')
        else:
            self.FEAT16 = torch.as_tensor(feat_z).to(torch.float16)
        self.PORT_COLS = list(port_cols)
        self.rows = {k: np.asarray(v, dtype=np.int64) for k, v in dict(train=train_rows, val=val_idx, test=test_idx).items()}
        self.current = dict(name=None, graph=None, targets=None, loader=None)
        log(f"간선 수 (정방향): 학습까지 {self.N['train']:,}  검증까지 {self.N['val']:,}  테스트까지 {self.N['test']:,}")

    @staticmethod
    def _rows_until(tx, window):
        _, end = L.parse_window(window)
        return int((tx['day'] <= pd.Timestamp(end)).sum())

    @property
    def edge_dim(self):
        return self.FEAT16.shape[1]

    def build_graph(self, n_rows, tx_rows):
        """tx 앞 n_rows 건으로 hetero 그래프 + target(tx_rows) 의 to 간선 위치. 타입마다 도착 노드순 정렬 (is_sorted=True 용).
        간선 속성은 edge_index 와 orig(tx 행 번호) 뿐. 텐서를 미리 할당하고 제자리에서 채워 임시 복사본을 안 만든다"""
        import torch
        from torch_geometric.data import HeteroData
        t0 = time.time(); m = n_rows
        data = HeteroData(); data['node'].x = torch.ones(self.NUM_NODES, 1); data['node'].num_nodes = self.NUM_NODES
        src = torch.as_tensor(self.SRC[:m]); dst = torch.as_tensor(self.DST[:m])
        # to: u → v, col = dst 순
        perm = torch.argsort(dst.to(torch.long))
        ei = torch.empty(2, m, dtype=torch.long); ei[0] = src[perm]; ei[1] = dst[perm]
        data[TO].edge_index, data[TO].orig, data[TO].y = ei, perm.to(torch.int32), torch.as_tensor(self.Y[:m])[perm]
        fwd_pos = torch.empty(m, dtype=torch.long); fwd_pos[perm] = torch.arange(m)
        targets = fwd_pos[torch.as_tensor(tx_rows)]
        assert (data[TO].y[targets].numpy() == self.Y[tx_rows]).all(), 'target 위치 오류'
        del perm, fwd_pos, ei
        # rev_to: v → u, col = src 순 (port 열 교환은 배치 변환에서)
        perm = torch.argsort(src.to(torch.long))
        ei = torch.empty(2, m, dtype=torch.long); ei[0] = dst[perm]; ei[1] = src[perm]
        data[REV].edge_index, data[REV].orig = ei, perm.to(torch.int32)
        del perm, ei, src, dst
        log(f'그래프 to {m:,} + rev_to {m:,} 간선 (정렬됨)  {time.time()-t0:.0f}s')
        return data, targets

    def make_loader(self, graph, targets, shuffle):
        from torch_geometric.loader import LinkNeighborLoader
        cfg = self.cfg
        return LinkNeighborLoader(
            graph, num_neighbors={TO: cfg.num_neighbors, REV: cfg.num_neighbors},
            edge_label_index=(TO, graph[TO].edge_index[:, targets]), edge_label=graph[TO].y[targets],
            batch_size=cfg.batch_size if shuffle else cfg.eval_batch_size, shuffle=shuffle,
            transform=AddEgoIdsAndTargetAttr(self.FEAT16, self.PORT_COLS, graph[TO].orig[targets]),
            is_sorted=True, num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)

    def use(self, name):
        """name 그래프('train' | 'val' | 'test') 와 loader 를 올린다. 이미 올라와 있으면 그대로. loader 를 전역에 따로 담지 말 것"""
        import torch
        if self.current['name'] == name:
            return self.current['loader']
        self.current.update(name=None, graph=None, targets=None, loader=None); trim()
        graph, targets = self.build_graph(self.N[name], self.rows[name])
        if name == 'train' and self.cfg.smoke:
            targets = targets[torch.randperm(len(targets))[:50_000]]
        loader = self.make_loader(graph, targets, shuffle=(name == 'train'))
        self.current.update(name=name, graph=graph, targets=targets, loader=loader); trim()
        return loader


class AddEgoIdsAndTargetAttr:
    """배치 변환
    1) ego ID 열을 x 에 붙임
    2) 엣지 피처를 FEAT16[orig] 로 꺼냄 (그래프에는 tx 행 번호만 있음). rev_to 는 in_port ↔ out_port 교환
    3) seed 간선 원 피처를 batch.target_attr 로 (target_rows = seed 들의 tx 행 번호, input_id 순)
    4) uint8 라벨 → long"""
    def __init__(self, feat16, port_cols, target_rows):
        self.F, self.port_cols, self.target_rows = feat16, list(port_cols), target_rows

    def rows(self, idx):
        """F[idx] 를 float32 텐서로. F 가 torch 텐서(RAM)든 numpy memmap(디스크)이든 같은 결과"""
        import torch
        if isinstance(self.F, torch.Tensor):
            return self.F[idx.long()].float()
        return torch.from_numpy(self.F[idx.numpy()].astype(np.float32))

    def __call__(self, batch):
        import torch
        ego = torch.zeros(batch['node'].num_nodes, 1); ego[batch[TO].edge_label_index.flatten()] = 1
        batch['node'].x = torch.cat([batch['node'].x, ego], 1)
        batch[TO].edge_attr = self.rows(batch[TO].orig)
        rev = self.rows(batch[REV].orig); rev[:, self.port_cols] = rev[:, self.port_cols[::-1]]
        batch[REV].edge_attr = rev
        batch.target_attr = self.rows(self.target_rows[batch[TO].input_id])
        batch[TO].edge_label = batch[TO].edge_label.long()
        return batch


# ============================================================================ 5. 모델
def make_gine(edge_dim, cfg, edge_updates):
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch_geometric.nn import GINEConv, BatchNorm

    class GINe(nn.Module):
        """Multi-GNN GINe + to_hetero(aggr='mean'). 간선 타입 fwd(to) / rev(rev_to) 각각 GINEConv, 출력 평균"""
        def __init__(self, num_features, edge_dim, n_hidden, num_gnn_layers, n_classes=2, edge_updates=False, dropout=0.0, final_dropout=0.5):
            super().__init__()
            self.n_hidden, self.num_gnn_layers, self.edge_updates, self.dropout = n_hidden, num_gnn_layers, edge_updates, dropout
            self.node_emb = nn.Linear(num_features, n_hidden)
            self.edge_emb = nn.ModuleDict({t: nn.Linear(edge_dim, n_hidden) for t in ('fwd', 'rev')})
            self.convs = nn.ModuleList(); self.emlps = nn.ModuleList(); self.batch_norms = nn.ModuleList()
            for _ in range(num_gnn_layers):
                self.convs.append(nn.ModuleDict({t: GINEConv(nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden)),
                                                             edge_dim=n_hidden) for t in ('fwd', 'rev')}))
                if edge_updates:
                    self.emlps.append(nn.ModuleDict({t: nn.Sequential(nn.Linear(3 * n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden)) for t in ('fwd', 'rev')}))
                self.batch_norms.append(BatchNorm(n_hidden))
            self.mlp = nn.Sequential(nn.Linear(n_hidden * 3, 50), nn.ReLU(), nn.Dropout(final_dropout),
                                     nn.Linear(50, 25), nn.ReLU(), nn.Dropout(final_dropout), nn.Linear(25, n_classes))

        def forward(self, x, ei, ea, target_index, target_attr):
            x = self.node_emb(x)
            ea = {t: self.edge_emb[t](ea[t]) for t in ('fwd', 'rev')}
            target_emb = self.edge_emb['fwd'](target_attr)
            for i in range(self.num_gnn_layers):
                h = (self.convs[i]['fwd'](x, ei['fwd'], ea['fwd']) + self.convs[i]['rev'](x, ei['rev'], ea['rev'])) / 2
                x = (x + F.relu(self.batch_norms[i](h))) / 2
                x = F.dropout(x, self.dropout, self.training)
                if self.edge_updates:
                    for t in ('fwd', 'rev'):
                        s, d = ei[t]; ea[t] = ea[t] + self.emlps[i][t](torch.cat([x[s], x[d], ea[t]], -1)) / 2
            h = torch.cat([x[target_index[0]], x[target_index[1]]], 1).relu()
            return self.mlp(torch.cat([h, target_emb], 1))

    return GINe(2, edge_dim, cfg.n_hidden, cfg.n_gnn_layers, edge_updates=edge_updates, dropout=cfg.dropout, final_dropout=cfg.final_dropout)


def forward_batch(model, batch, device):
    batch = batch.to(device)
    logits = model(batch['node'].x, {'fwd': batch[TO].edge_index, 'rev': batch[REV].edge_index},
                   {'fwd': batch[TO].edge_attr, 'rev': batch[REV].edge_attr}, batch[TO].edge_label_index, batch.target_attr)
    return logits, batch[TO].edge_label


class InfiniteLoader:
    """셔플된 loader 를 끝까지 읽으면 다시 섞어 이어 준다 (FraudGT LoaderWrapper 의 train 동작)"""
    def __init__(self, loader):
        self.loader, self.it = loader, iter(loader)

    def next(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader); return next(self.it)


def predict_loader(model, loader, device):
    import torch, torch.nn.functional as F
    model.eval(); out = []
    with torch.no_grad():
        for b in loader:
            logits, _ = forward_batch(model, b, device)
            out.append(F.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


# ============================================================================ 6. 학습 · 테스트
def train_gnn(name, store, cfg, val_df, out_dir, device, edge_updates=False, results=None):
    """01 의 train_gnn 그대로. 검증은 eval_period 에폭마다 (그래프 교체 포함), 검증 AP 최고 체크포인트 저장"""
    import torch, torch.nn as nn
    from tqdm.auto import tqdm
    torch.manual_seed(cfg.seed)
    model = make_gine(store.edge_dim, cfg, edge_updates).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cfg.class_weight, device=device))
    history, best_ap, stale, stream = [], -np.inf, 0, None
    ckpt = out_dir / f'{name}.pt'; t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        if store.current['name'] != 'train' or stream is None:
            stream = InfiniteLoader(store.use('train'))
        model.train(); losses = []; te = time.time()
        for _ in tqdm(range(cfg.iter_per_epoch), desc=f'{name} epoch {epoch}', leave=False):
            logits, labels = forward_batch(model, stream.next(), device)
            loss = loss_fn(logits, labels)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), epoch_sec=round(time.time() - te))
        if epoch % cfg.eval_period == 0 or epoch == cfg.epochs:
            stream = None
            m = evaluate_sample(val_df, predict_loader(model, store.use('val'), device)); row.update(m)
            print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  AP {m['AP']:.4f}  F1 {m['F1']:.4f}  F1@0.5 {m['F1_at_0.5']:.4f}  "
                  f"recall@1% {m['recall_at_1pct']:.4f}  ({time.time() - te:.0f}s)")
            if m['AP'] > best_ap:
                best_ap, stale = m['AP'], 0; torch.save(model.state_dict(), ckpt)
            else:
                stale += 1
        else:
            print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  ({row['epoch_sec']}s)")
        history.append(row)
        if stale >= cfg.patience:
            break
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    hist = pd.DataFrame(history); hist.to_csv(out_dir / f'{name}_history.csv', index=False)
    best = hist.loc[hist['AP'].idxmax()]
    rec = dict(val={k: float(best[k]) for k in VAL_KEYS}, test=None, best_epoch=int(best['epoch']), epochs_run=len(hist),
               train_sec=round(time.time() - t_start), targets_seen=int(best['epoch']) * cfg.iter_per_epoch * cfg.batch_size,
               gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == 'cuda' else None)
    if results is not None:
        results[name] = rec
    return model, hist


def test_gnn(name, store, cfg, test_df, out_dir, device, edge_updates, results):
    """저장된 체크포인트로 테스트 구간 전체 추론. 점수 parquet 저장"""
    import torch
    if not (out_dir / f'{name}.pt').exists():
        print(name, '체크포인트 없음 — 건너뜀'); return None
    if name not in results:        # 학습 셀을 건너뛰고 왔으면 history.csv 에서 검증 결과 복원
        hist = pd.read_csv(out_dir / f'{name}_history.csv'); best = hist.loc[hist['AP'].idxmax()]
        results[name] = dict(val={k: float(best[k]) for k in VAL_KEYS}, test=None, best_epoch=int(best['epoch']), epochs_run=len(hist),
                             train_sec=int(hist['epoch_sec'].sum()), targets_seen=int(best['epoch']) * cfg.iter_per_epoch * cfg.batch_size, gpu_peak_gb=None)
    model = make_gine(store.edge_dim, cfg, edge_updates).to(device)
    model.load_state_dict(torch.load(out_dir / f'{name}.pt', map_location=device, weights_only=True))
    t0 = time.time(); score = predict_loader(model, store.use('test'), device)
    pd.DataFrame(dict(edge_id=test_df.edge_id, y=test_df.y, score=score)).to_parquet(out_dir / f'{name}_test_scores.parquet', index=False)
    results[name]['test'] = evaluate_sample(test_df, score)
    print(name, f'({(time.time()-t0)/60:.0f}분)', {k: round(v, 4) for k, v in results[name]['test'].items()})
    return score


def save_results(out_dir, cfg, results, features):
    metric_cols = ['AP', 'F1', 'F1_at_0.5', 'precision', 'recall', 'recall_at_1pct', 'unmatched_recall_at_1pct', 'ECE']
    rows = []
    for name, r in results.items():
        for split in ('val', 'test'):
            if r.get(split) is None:
                continue
            rows.append(dict(model=name, split=split, **{k: r[split][k] for k in metric_cols}, train_sec=r['train_sec']))
    compare = pd.DataFrame(rows).set_index(['model', 'split']).round(4)
    (out_dir / 'results.json').write_text(json.dumps(dict(
        dataset=cfg.dataset, smoke=cfg.smoke, features=features, windows=dict(train=cfg.train_window, val=cfg.val_window, test=cfg.test_window),
        val_neg_rate=cfg.val_neg_rate, scale_pos_weight=cfg.scale_pos_weight, class_weight=cfg.class_weight,
        num_neighbors=cfg.num_neighbors, batch_size=cfg.batch_size, iter_per_epoch=cfg.iter_per_epoch, results=results,
    ), ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    compare.to_csv(out_dir / 'compare.csv', encoding='utf-8-sig')
    return compare


# ============================================================================ 7. 분석 (0단계)
PATTERN_ORDER = ['FAN-OUT', 'FAN-IN', 'GATHER-SCATTER', 'SCATTER-GATHER', 'CYCLE', 'RANDOM', 'BIPARTITE', 'STACK', 'UNMATCHED']
DEGREE_BINS = [(1, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 10**9)]


def load_scores(path):
    """*_test_scores.parquet → DataFrame(edge_id, y, score)"""
    return pd.read_parquet(path)


BUDGETS = (0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0)   # 알림 예산 (테스트 전체 상위 비율). 0.1% ≈ 양성 수 규모 ≈ argmax 운영점, 1% = 논문 관행, 1.0 = 전체


def cut_name(b):
    return 'all' if b >= 1 else f'r@{b*100:g}%'


def cut_thresholds(score, budgets=BUDGETS, argmax=True):
    """운영점(컷) 이름 → 임계값. 'r@0.1%' … 는 예산 컷, 'all' 은 전부 알림, 'argmax' 는 확률 0.5 (Multi-GNN 부록 Table 13 의 recall 과 같은 기준)"""
    cuts = {cut_name(b): (float(score.min()) if b >= 1 else float(np.quantile(score, 1 - b))) for b in budgets}
    if argmax:
        cuts['argmax'] = 0.5
    return cuts


def cut_table(scores, budgets=BUDGETS, argmax=True):
    """컷별 임계값·알림 수·TP·FP·FN·precision·recall 한 표 (confusion_breakdown 의 matrix 만)"""
    return confusion_breakdown(scores, {}, budgets, argmax)['matrix']


def group_by_cut(scores, labels, budgets=BUDGETS, argmax=True, min_pos=1):
    """그룹(라벨 배열) × 컷 → recall (그룹 안 양성 중 잡은 비율) 과 precision (그룹 안 알림 중 양성 비율).
    패턴이면 양성만 있는 그룹이라 recall 만, 차수 구간이면 둘 다 의미 있다"""
    y = scores['y'].to_numpy().astype(bool); sc = scores['score'].to_numpy(); lab = np.asarray(labels)
    cuts = cut_thresholds(sc, budgets, argmax)
    rec, prec, n_pos = {}, {}, {}
    for g in pd.unique(lab):
        m = lab == g; yg = y[m]; sg = sc[m]; n_pos[g] = int(yg.sum())
        if n_pos[g] < min_pos:
            continue
        rec[g] = {k: float((yg & (sg >= t)).sum() / n_pos[g]) for k, t in cuts.items()}
        prec[g] = {k: float((yg & (sg >= t)).sum() / max((sg >= t).sum(), 1)) for k, t in cuts.items()}
    return dict(recall=pd.DataFrame(rec).T.assign(n_pos=pd.Series(n_pos)).round(4),
                precision=pd.DataFrame(prec).T.assign(n_pos=pd.Series(n_pos)).round(4))


def pattern_label(scores):
    """scores 의 양성 행에 pattern_type 을 붙인다. patterns.parquet 에 없는 양성 = UNMATCHED"""
    pat = pd.read_parquet(L.BASE / 'patterns.parquet', columns=['edge_id', 'pattern_type']).dropna().drop_duplicates('edge_id')
    pos = scores[scores.y == 1].merge(pat, on='edge_id', how='left')
    pos['pattern_type'] = pos['pattern_type'].fillna('UNMATCHED')
    return pos


def pattern_array(scores):
    """scores 전체 행 순서의 패턴 라벨 배열 (정상 = 'NORMAL'). confusion_breakdown 의 pattern 축용"""
    pat = pd.read_parquet(L.BASE / 'patterns.parquet', columns=['edge_id', 'pattern_type']).dropna().drop_duplicates('edge_id')
    lab = scores[['edge_id', 'y']].merge(pat, on='edge_id', how='left')['pattern_type']
    lab = lab.where(scores['y'].to_numpy() == 0, lab.fillna('UNMATCHED'))
    return lab.fillna('NORMAL').to_numpy()


def pattern_recall(scores, budget=0.01, budgets=BUDGETS, argmax=True):
    """패턴 유형별 recall. 열: n, recall(=budget 컷, 기존 호환), r@0.1% … r@5% (예산을 줄이면 어느 패턴이 먼저 떨어지나), r@argmax, mean_score.
    같은 패턴 안의 거래는 모두 양성이므로 recall 만 본다"""
    pos = pattern_label(scores)
    cuts = {'recall': float(np.quantile(scores['score'].to_numpy(), 1 - budget))}
    cuts.update(cut_thresholds(scores['score'].to_numpy(), budgets, argmax))
    for k, thr in cuts.items():
        pos[k] = pos['score'] >= thr
    g = pos.groupby('pattern_type').agg(n=('score', 'size'), **{k: (k, 'mean') for k in cuts}, mean_score=('score', 'mean'))
    g = g.reindex([p for p in PATTERN_ORDER if p in g.index] + [p for p in g.index if p not in PATTERN_ORDER])
    g.loc['ALL'] = dict(n=len(pos), **{k: pos[k].mean() for k in cuts}, mean_score=pos['score'].mean())
    g.attrs['thresholds'] = cuts
    return g.round(4)


def recipient_indegree(con, cfg):
    """수신 계좌 in-degree = 테스트 구간 끝까지의 거래에서 dst 별 받은 거래 수 (SALT-GNN: recipient context)"""
    _, end = L.parse_window(cfg.test_window)
    df = con.execute(f"SELECT dst AS node_id, count(*) AS in_degree FROM read_parquet({L.q(cfg.tx_path)}) WHERE day <= DATE {L.q(end)} GROUP BY dst").df()
    return df.set_index('node_id')['in_degree']


def degree_stratified(scores, dst_of_edge, in_degree, budget=0.01):
    """수신 계좌 in-degree 구간별 지표 (SALT-GNN 층화 평가). 학습·선택은 그대로, 테스트 예측만 구간으로 나눈다.
    dst_of_edge: scores 와 같은 순서의 dst 배열. in_degree: node_id → in-degree Series"""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    s = scores.copy(); s['deg'] = in_degree.reindex(dst_of_edge).to_numpy()
    thr = np.quantile(s['score'].to_numpy(), 1 - budget)
    rows = []
    for lo, hi in DEGREE_BINS:
        g = s[(s.deg >= lo) & (s.deg <= hi)]
        y = g.y.to_numpy(); sc = g.score.to_numpy(); n_pos = int(y.sum())
        row = dict(bin=f'[{lo}-{hi}]' if hi < 10**9 else f'[{lo}+]', n=len(g), n_pos=n_pos, pos_rate=round(y.mean(), 5) if len(g) else np.nan)
        if n_pos > 0 and n_pos < len(g):
            p, r, _ = precision_recall_curve(y, sc); f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
            pred = sc >= 0.5; tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
            row.update(AP=average_precision_score(y, sc), F1=f1.max(), F1_at_0_5=2 * tp / max(2 * tp + fp + fn, 1e-12),
                       recall_at_budget=((sc >= thr) & (y == 1)).sum() / n_pos, recall_argmax=tp / n_pos)
        rows.append(row)
    return pd.DataFrame(rows).set_index('bin').round(4)


def degree_bin_of(dst_of_edge, in_degree):
    """간선별 수신 계좌 in-degree 구간 라벨 (confusion_breakdown 의 축으로 쓴다)"""
    deg = in_degree.reindex(dst_of_edge).to_numpy()
    labels = [f'[{lo}-{hi}]' if hi < 10**9 else f'[{lo}+]' for lo, hi in DEGREE_BINS]
    return pd.cut(deg, [lo - 0.5 for lo, _ in DEGREE_BINS] + [np.inf], labels=labels).astype(str)


def confusion_breakdown(scores, groups, budgets=BUDGETS, argmax=True):
    """컷마다 2×2 혼동행렬 + "뭐를 헷갈렸나" 를 축별로 쪼갠 표.
    groups: {축 이름: scores 와 같은 순서의 라벨 배열}. 예) pattern(양성만 의미 있음), degree_bin, payment_format, currency
    돌려주는 것:
      matrix  — 컷별 TP/FP/FN/TN, alerts, precision, recall
      fn[axis] — 컷별 놓친 양성(FN) 수와 FN 안 비중, 그 축의 전체 양성 비중 (비중이 양성 비중보다 크면 그 그룹을 유독 놓침)
      fp[axis] — 컷별 잘못 잡은 정상(FP) 수와 FP 안 비중, 그 축의 전체 정상 비중 (비중이 정상 비중보다 크면 그 그룹에 알림이 몰림)"""
    y = scores['y'].to_numpy().astype(bool); sc = scores['score'].to_numpy()
    cuts = cut_thresholds(sc, budgets, argmax)
    mat, fn_tabs, fp_tabs = [], {}, {}
    for k, thr in cuts.items():
        pred = sc >= thr
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
        mat.append(dict(cut=k, threshold=round(thr, 4), alerts=tp + fp, TP=tp, FP=fp, FN=fn, TN=tn,
                        precision=round(tp / max(tp + fp, 1), 4), recall=round(tp / max(tp + fn, 1), 4)))
    for axis, lab in groups.items():
        lab = pd.Series(np.asarray(lab), index=scores.index)
        fn_t = pd.DataFrame({'pos_share': lab[y].value_counts(normalize=True)})
        fp_t = pd.DataFrame({'neg_share': lab[~y].value_counts(normalize=True)})
        for k, thr in cuts.items():
            pred = sc >= thr
            fn_c = lab[y & ~pred].value_counts(); fp_c = lab[~y & pred].value_counts()
            fn_t[f'FN {k}'] = fn_c; fn_t[f'FN% {k}'] = fn_c / max(fn_c.sum(), 1)
            fp_t[f'FP {k}'] = fp_c; fp_t[f'FP% {k}'] = fp_c / max(fp_c.sum(), 1)
        fn_tabs[axis] = fn_t.fillna(0).sort_values('pos_share', ascending=False).round(4)
        fp_tabs[axis] = fp_t.fillna(0).sort_values('neg_share', ascending=False).round(4)
    return dict(matrix=pd.DataFrame(mat).set_index('cut'), fn=fn_tabs, fp=fp_tabs)


def analyze(name, scores_path, cfg, con, test_dst):
    """한 모델의 테스트 점수로 두 표를 뽑아 돌려준다"""
    sc = load_scores(scores_path)
    pr = pattern_recall(sc)
    ds = degree_stratified(sc, test_dst, recipient_indegree(con, cfg))
    return pr.assign(model=name), ds.assign(model=name)


# ============================================================================ 8. 그림 (결과 절)
def plot_pr(scores_by_model, path=None, mark=None):
    """모델별 PR 곡선 (범례에 AP). mark: (recall, 라벨) 세로선, 예) argmax 운영점"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, sc in scores_by_model.items():
        y = sc['y'].to_numpy(); s = sc['score'].to_numpy()
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r[::50], p[::50], lw=1.5, label=f'{name} (AP {average_precision_score(y, s):.3f})')   # 2,700 만 점은 50 개 중 1 개만 그려도 곡선 동일
    if mark is not None:
        ax.axvline(mark[0], ls='--', c='gray', lw=1, label=mark[1])
    ax.set_xlabel('recall'); ax.set_ylabel('precision'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(loc='upper right')
    plt.tight_layout()
    if path: plt.savefig(path, dpi=120)
    plt.show()


def plot_confusion_grid(scores, cuts=('r@0.1%', 'r@1%', 'r@10%', 'argmax'), title='', path=None, budgets=BUDGETS):
    """컷마다 2×2 혼동행렬을 나란히. 색 = 행 비율 (세탁 행 = recall), 숫자 = 건수. TN 이 2,700 만이라 건수 색은 의미 없어 행 정규화"""
    import matplotlib.pyplot as plt
    mat = cut_table(scores, budgets).loc[list(cuts)]
    fig, axes = plt.subplots(1, len(cuts), figsize=(3.2 * len(cuts), 3.4))
    for ax, (k, m) in zip(np.atleast_1d(axes), mat.iterrows()):
        cm = np.array([[m.TN, m.FP], [m.FN, m.TP]], dtype=np.int64); rate = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rate, cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]:,}\n({rate[i, j]:.1%})', ha='center', va='center', fontsize=8, color='white' if rate[i, j] > 0.6 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['pred normal', 'pred laund.'], fontsize=8); ax.set_yticks([0, 1]); ax.set_yticklabels(['normal', 'laund.'], fontsize=8)
        ax.set_title(f'{k} (thr {m.threshold:.3f})\nP {m.precision:.3f}  R {m.recall:.3f}', fontsize=9)
    fig.suptitle(title); plt.tight_layout()
    if path: plt.savefig(path, dpi=120)
    plt.show()


def plot_heatmap(df, title='', path=None, fmt='{:.2f}', figsize=None, vmax=1.0):
    """행 × 열 표를 색 + 숫자로. 패턴 × 컷 recall, 차수 구간 × 컷 precision 등에 쓴다"""
    import matplotlib.pyplot as plt
    a = df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=figsize or (1.0 * df.shape[1] + 2, 0.45 * df.shape[0] + 1.2))
    ax.imshow(a, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index, fontsize=8)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if not np.isnan(a[i, j]):
                ax.text(j, i, fmt.format(a[i, j]), ha='center', va='center', fontsize=7, color='white' if a[i, j] > 0.6 * vmax else 'black')
    ax.set_title(title, fontsize=10); plt.tight_layout()
    if path: plt.savefig(path, dpi=120)
    plt.show()
