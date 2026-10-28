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
    """지운 메모리를 OS 에 돌려주고(glibc 는 안 돌려줄 때가 많다) 현재 RSS 를 찍는다"""
    gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)
    print(f'RSS {psutil.Process().memory_info().rss / 1e9:.1f} GB {label}')


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


def load_tx(con, cfg):
    """거래 전체 (edge_id 순): edge_id, src, dst, day, 기본 피처 4, y.  문자열 열은 pandas 에 올리지 않는다 (코드 테이블 조인)"""
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


def attach_parquet(tx, path, cols):
    """edge_id 순으로 저장된 피처 parquet 의 열들을 tx 에 붙인다 (행 순서가 같은지 확인). DuckDB 조인은 1.8억 × 1.8억이라 메모리 초과"""
    import pyarrow.parquet as pq
    tbl = pq.read_table(path, columns=['edge_id'] + list(cols))
    assert (tbl.column('edge_id').to_numpy() == tx['edge_id'].to_numpy()).all(), f'{path.name} 행 순서가 tx 와 다름'
    for c in cols:
        tx[c] = tbl.column(c).to_numpy()
    del tbl; gc.collect()
    return tx


def window_mask(df, window):
    a, b = L.parse_window(window)
    return ((df['day'] >= pd.Timestamp(a)) & (df['day'] <= pd.Timestamp(b))).to_numpy()


def znorm_features(tx, cols, in_train):
    """tx 의 cols 를 열 하나씩 float32 배열로 뽑아 학습 구간 통계로 z-norm. 뽑은 열은 tx 에서 떨어뜨린다.
    (tx[cols].to_numpy() 는 dtype 이 섞여 float64 로 먼저 합쳐 1.8억 × 6 이면 8.6GB → 열 단위로)"""
    feat_z = np.empty((len(tx), len(cols)), np.float32)
    mu = np.empty(len(cols), np.float32); sd = np.empty(len(cols), np.float32)
    for j, col in enumerate(cols):
        feat_z[:, j] = tx[col].to_numpy()
        mu[j] = feat_z[in_train, j].mean(); sd[j] = feat_z[in_train, j].std() + 1e-6
        feat_z[:, j] -= mu[j]; feat_z[:, j] /= sd[j]
    tx.drop(columns=list(cols), inplace=True); gc.collect()
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
def train_trees(X_train, y_train, X_val, y_val, w_val, X_test, val_df, test_df, cfg, results, n_jobs=8):
    import lightgbm as lgb, xgboost as xgb
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(n_estimators=cfg.tree_rounds, learning_rate=0.05, num_leaves=63, scale_pos_weight=cfg.scale_pos_weight,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=1.0, random_state=cfg.seed, n_jobs=n_jobs, verbose=-1)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_sample_weight=[w_val], eval_metric='average_precision',
                  callbacks=[lgb.early_stopping(cfg.tree_early_stop, verbose=False)])
    results['lightgbm'] = dict(val=evaluate_sample(val_df, lgb_model.predict_proba(X_val)[:, 1]),
                               test=evaluate_sample(test_df, lgb_model.predict_proba(X_test)[:, 1]),
                               best_iteration=int(lgb_model.best_iteration_ or cfg.tree_rounds), train_sec=round(time.time() - t0))
    log(f"lightgbm val AP {results['lightgbm']['val']['AP']:.4f}")
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(n_estimators=cfg.tree_rounds, learning_rate=0.05, max_depth=8, scale_pos_weight=cfg.scale_pos_weight,
                                  subsample=0.8, colsample_bytree=1.0, tree_method='hist', device='cpu',
                                  eval_metric='aucpr', early_stopping_rounds=cfg.tree_early_stop, random_state=cfg.seed, n_jobs=n_jobs)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val], verbose=False)
    results['xgboost'] = dict(val=evaluate_sample(val_df, xgb_model.predict_proba(X_val)[:, 1]),
                              test=evaluate_sample(test_df, xgb_model.predict_proba(X_test)[:, 1]),
                              best_iteration=int(xgb_model.best_iteration), train_sec=round(time.time() - t0))
    log(f"xgboost  val AP {results['xgboost']['val']['AP']:.4f}")
    return lgb_model, xgb_model


# ============================================================================ 4. 그래프
class GraphStore:
    """그래프 재료(SRC, DST, Y, FEAT16)를 들고 학습·검증·테스트 그래프를 **한 번에 하나만** 올린다 (01 의 use_graph).
    FEAT16 열 순서: 기본 4 + in_port + out_port + (단계별 추가 피처). rev_to 간선은 port 두 열을 바꿔 붙인다."""

    def __init__(self, tx, feat_z, cfg, train_rows, val_idx, test_idx, port_cols):
        import torch
        self.cfg = cfg
        self.NUM_NODES = int(max(tx['src'].max(), tx['dst'].max())) + 1
        self.N = {k: self._rows_until(tx, w) for k, w in (('train', cfg.train_window), ('val', cfg.val_window), ('test', cfg.test_window))}
        self.SRC = tx['src'].to_numpy().copy(); self.DST = tx['dst'].to_numpy().copy(); self.Y = tx['y'].to_numpy().astype(np.uint8)
        self.FEAT16 = torch.as_tensor(feat_z).to(torch.float16)
        self.PORT_COLS = list(port_cols)
        self.rows = dict(train=train_rows, val=val_idx, test=test_idx)
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
        텐서를 미리 할당하고 제자리에서 채워 임시 복사본을 안 만든다"""
        import torch
        from torch_geometric.data import HeteroData
        t0 = time.time(); m = n_rows; F = self.FEAT16
        data = HeteroData(); data['node'].x = torch.ones(self.NUM_NODES, 1); data['node'].num_nodes = self.NUM_NODES
        src = torch.as_tensor(self.SRC[:m]); dst = torch.as_tensor(self.DST[:m])
        perm = torch.argsort(dst.to(torch.long))
        ei = torch.empty(2, m, dtype=torch.long); ei[0] = src[perm]; ei[1] = dst[perm]
        attr = torch.empty(m, F.shape[1], dtype=torch.float16); torch.index_select(F, 0, perm, out=attr)
        data[TO].edge_index, data[TO].edge_attr, data[TO].y = ei, attr, torch.as_tensor(self.Y[:m])[perm]
        fwd_pos = torch.empty(m, dtype=torch.long); fwd_pos[perm] = torch.arange(m)
        targets = fwd_pos[torch.as_tensor(np.asarray(tx_rows))]
        assert (data[TO].y[targets].numpy() == self.Y[np.asarray(tx_rows)]).all(), 'target 위치 오류'
        del perm, fwd_pos, ei, attr
        perm = torch.argsort(src.to(torch.long))
        ei = torch.empty(2, m, dtype=torch.long); ei[0] = dst[perm]; ei[1] = src[perm]
        attr = torch.empty(m, F.shape[1], dtype=torch.float16); torch.index_select(F, 0, perm, out=attr)
        attr[:, self.PORT_COLS] = attr[:, self.PORT_COLS[::-1]]
        data[REV].edge_index, data[REV].edge_attr = ei, attr
        del perm, ei, attr, src, dst
        log(f'그래프 to {m:,} + rev_to {m:,} 간선 (정렬됨)  {time.time()-t0:.0f}s')
        return data, targets

    def make_loader(self, graph, targets, shuffle):
        from torch_geometric.loader import LinkNeighborLoader
        cfg = self.cfg
        return LinkNeighborLoader(
            graph, num_neighbors={TO: cfg.num_neighbors, REV: cfg.num_neighbors},
            edge_label_index=(TO, graph[TO].edge_index[:, targets]), edge_label=graph[TO].y[targets],
            batch_size=cfg.batch_size, shuffle=shuffle, transform=AddEgoIdsAndTargetAttr(graph, targets),
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
    """1) ego ID 열을 x 에 붙임  2) seed 간선 원 피처를 batch.target_attr 로  3) float16/uint8 → float32/long"""
    def __init__(self, graph, targets):
        self.edge_attr, self.targets = graph[TO].edge_attr, targets

    def __call__(self, batch):
        import torch
        ego = torch.zeros(batch['node'].num_nodes, 1); ego[batch[TO].edge_label_index.flatten()] = 1
        batch['node'].x = torch.cat([batch['node'].x, ego], 1)
        batch[TO].edge_attr = batch[TO].edge_attr.float(); batch[REV].edge_attr = batch[REV].edge_attr.float()
        batch.target_attr = self.edge_attr[self.targets[batch[TO].input_id]].float()
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


def pattern_recall(scores, budget=0.01):
    """패턴 유형별 recall@budget. 임계값은 테스트 전체 상위 budget 비율 (알림 예산) 하나로 고정.
    patterns.parquet 에 없는 양성 = UNMATCHED. 같은 패턴 안의 거래는 모두 양성이므로 recall 만 본다"""
    pat = pd.read_parquet(L.BASE / 'patterns.parquet', columns=['edge_id', 'pattern_type']).dropna().drop_duplicates('edge_id')
    thr = np.quantile(scores['score'].to_numpy(), 1 - budget)
    pos = scores[scores.y == 1].merge(pat, on='edge_id', how='left')
    pos['pattern_type'] = pos['pattern_type'].fillna('UNMATCHED')
    pos['hit'] = pos['score'] >= thr
    g = pos.groupby('pattern_type').agg(n=('hit', 'size'), recall=('hit', 'mean'), mean_score=('score', 'mean'))
    g = g.reindex([p for p in PATTERN_ORDER if p in g.index] + [p for p in g.index if p not in PATTERN_ORDER])
    g.loc['ALL'] = dict(n=len(pos), recall=pos['hit'].mean(), mean_score=pos['score'].mean())
    g.attrs['threshold'] = float(thr)
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
                       recall_at_budget=((sc >= thr) & (y == 1)).sum() / n_pos)
        rows.append(row)
    return pd.DataFrame(rows).set_index('bin').round(4)


def analyze(name, scores_path, cfg, con, test_dst):
    """한 모델의 테스트 점수로 두 표를 뽑아 돌려준다"""
    sc = load_scores(scores_path)
    pr = pattern_recall(sc)
    ds = degree_stratified(sc, test_dst, recipient_indegree(con, cfg))
    return pr.assign(model=name), ds.assign(model=name)
