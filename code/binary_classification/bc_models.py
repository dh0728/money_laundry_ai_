"""2단계 모델 비교용 GNN 백본 — bc_lib.make_gine 과 같은 입·출력 규약으로 PNA · GAT · TransConv · SALT-Trans 를 만든다.

규약 (bc_lib.forward_batch 가 부르는 꼴):
    model(x, ei={'fwd','rev'}, ea={'fwd','rev'}, target_index, target_attr) → logits (n_target, 2)
  - 간선 타입마다(to / rev_to) conv 하나씩 두고 출력을 평균 (to_hetero aggr='mean' 과 같은 꼴, make_gine 과 동일)
  - 층마다 x = (x + relu(BN(h))) / 2 (Multi-GNN 의 residual 평균). 분류 헤드도 Multi-GNN 과 같다 ([x_u ‖ x_v] relu ‖ edge_emb(target))
  - 하이퍼파라미터는 MODEL_SETTINGS (plan.md 2단계 표 = Multi-GNN model_settings / SALT 부록 I). 모델별 재튜닝 없음

SALT 층 (SALT_GNN.md 식 4~11):
  통계 채널  = PNAConv(mean·std·min·max × identity·amplification·attenuation, deg 히스토그램) — PNAConv 의 post MLP 가 [h_prev ‖ 통계] → MLP (식 5)
  attention 채널 = TransformerConv(root_weight=False) — 이웃 가중 평균 (식 6~8)
  융합       = MLP([h_prev ‖ h_stat ‖ h_att]) (식 9~10) → BN·relu 후 h_prev 와 평균 (식 11)
"""
from __future__ import annotations
import numpy as np

# plan.md 2단계 표. heads 는 attention 계열만 사용. class_weight 는 CE 양성 가중치
MODEL_SETTINGS = {
    'gine':       dict(lr=0.0062, n_hidden=66,  class_weight=[1.0, 6.27], dropout=0.010, final_dropout=0.105),              # Multi-GNN 'gin'
    'gine128':    dict(lr=0.0062, n_hidden=128, class_weight=[1.0, 6.27], dropout=0.010, final_dropout=0.105),              # 같은 설정, hidden 만 2배 (06 에서 트리와의 차이가 용량 탓인지)
    'pna':        dict(lr=0.0006, n_hidden=20,  class_weight=[1.0, 7.08], dropout=0.083, final_dropout=0.288),              # Multi-GNN 'pna'
    'gat':        dict(lr=0.006,  n_hidden=64,  class_weight=[1.0, 6.0],  dropout=0.009, final_dropout=0.1, heads=4),       # Multi-GNN 'gat'
    'trans':      dict(lr=0.001,  n_hidden=64,  class_weight=[1.0, 6.0],  dropout=0.009, final_dropout=0.1, heads=4),       # SALT 부록 I (v10 백본)
    'salt_trans': dict(lr=0.0012, n_hidden=20,  class_weight=[1.0, 7.0],  dropout=0.083, final_dropout=0.288, heads=4),     # SALT 부록 I (PNA 값 상속)
}
NEEDS_DEG = {'pna', 'salt_trans'}   # PNA 스케일러용 degree 히스토그램이 필요한 모델


def apply_settings(cfg, kind, seed):
    """cfg 를 복사해 모델 설정과 시드를 넣는다 (학습 프로토콜 — 배치·이웃·에폭·patience — 는 그대로)"""
    import dataclasses
    s = dict(MODEL_SETTINGS[kind]); heads = s.pop('heads', cfg.heads)
    return dataclasses.replace(cfg, seed=seed, heads=heads, **s)


def degree_histograms(store):
    """PNAConv 의 deg 인자 — 학습 그래프(학습 구간 끝까지의 간선) 에서 도착 노드 in-degree 분포. fwd(to) 는 dst 쪽, rev(rev_to) 는 src 쪽.
    Multi-GNN 과 같이 전체 학습 그래프 차수로 센다 (이웃 샘플링 전)"""
    import torch
    n = store.N['train']
    out = {}
    for t, arr in (('fwd', store.DST[:n]), ('rev', store.SRC[:n])):
        deg = np.bincount(arr.astype(np.int64), minlength=store.NUM_NODES)
        out[t] = torch.as_tensor(np.bincount(deg), dtype=torch.long)
    return out


def _mlp(i, h, o):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


def make_model(kind, edge_dim, cfg, store=None, deg=None):
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch_geometric.nn import GINEConv, GATConv, TransformerConv, PNAConv, BatchNorm

    if kind in NEEDS_DEG and deg is None:
        assert store is not None, f'{kind} 는 degree 히스토그램이 필요하다 (store 또는 deg)'
        deg = degree_histograms(store)
    h, heads, drop = cfg.n_hidden, cfg.heads, cfg.dropout
    assert h % heads == 0, f'n_hidden {h} 가 heads {heads} 로 나눠지지 않음'

    def conv_for(t):
        """간선 타입 t('fwd'|'rev') 하나의 메시지 전달 층. 입력·출력 모두 n_hidden, edge_dim=n_hidden (edge_emb 뒤)"""
        if kind in ('gine', 'gine128'):
            return GINEConv(_mlp(h, h, h), edge_dim=h)
        if kind == 'gat':
            return GATConv(h, h // heads, heads=heads, concat=True, dropout=drop, add_self_loops=True, edge_dim=h)
        if kind == 'trans':
            return TransformerConv(h, h // heads, heads=heads, concat=True, dropout=drop, edge_dim=h)
        if kind == 'pna':
            return PNAConv(h, h, aggregators=['mean', 'max', 'min', 'std'], scalers=['identity', 'amplification', 'attenuation'],
                           deg=deg[t], edge_dim=h, towers=5 if h % 5 == 0 else 1, pre_layers=1, post_layers=1, divide_input=False)
        if kind == 'salt_trans':
            return SALTLayer(h, heads, drop, deg[t])
        raise ValueError(kind)

    class SALTLayer(nn.Module):
        """통계 채널(PNA) + attention 채널(TransformerConv) 을 같은 메시지 집합에 돌려 [h_prev ‖ h_stat ‖ h_att] → MLP"""
        def __init__(self, h, heads, drop, deg_t):
            super().__init__()
            self.stat = PNAConv(h, h, aggregators=['mean', 'std', 'min', 'max'], scalers=['identity', 'amplification', 'attenuation'],
                                deg=deg_t, edge_dim=h, towers=1, pre_layers=1, post_layers=1, divide_input=False)       # 식 4~5 (post MLP 가 [x ‖ 12 통계] 를 받음)
            self.att = TransformerConv(h, h // heads, heads=heads, concat=True, dropout=drop, edge_dim=h, root_weight=False)  # 식 6~8 (자기 항 없음 — 융합이 맡음)
            self.fusion = _mlp(3 * h, h, h)                                                                             # 식 9~10

        def forward(self, x, edge_index, edge_attr):
            return self.fusion(torch.cat([x, self.stat(x, edge_index, edge_attr), self.att(x, edge_index, edge_attr)], -1))

    class HeteroNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_emb = nn.Linear(2, h)                                             # x = [1, ego]
            self.edge_emb = nn.ModuleDict({t: nn.Linear(edge_dim, h) for t in ('fwd', 'rev')})
            self.convs = nn.ModuleList([nn.ModuleDict({t: conv_for(t) for t in ('fwd', 'rev')}) for _ in range(cfg.n_gnn_layers)])
            self.bns = nn.ModuleList([BatchNorm(h) for _ in range(cfg.n_gnn_layers)])
            self.mlp = nn.Sequential(nn.Linear(h * 3, 50), nn.ReLU(), nn.Dropout(cfg.final_dropout),
                                     nn.Linear(50, 25), nn.ReLU(), nn.Dropout(cfg.final_dropout), nn.Linear(25, 2))

        def forward(self, x, ei, ea, target_index, target_attr):
            x = self.node_emb(x)
            ea = {t: self.edge_emb[t](ea[t]) for t in ('fwd', 'rev')}
            target_emb = self.edge_emb['fwd'](target_attr)
            for conv, bn in zip(self.convs, self.bns):
                hh = (conv['fwd'](x, ei['fwd'], ea['fwd']) + conv['rev'](x, ei['rev'], ea['rev'])) / 2
                x = (x + F.relu(bn(hh))) / 2                                            # residual 평균 (SALT 식 11 = Multi-GNN)
                x = F.dropout(x, drop, self.training)
            out = torch.cat([x[target_index[0]], x[target_index[1]]], 1).relu()
            return self.mlp(torch.cat([out, target_emb], 1))

    return HeteroNet()


def n_params(model):
    return sum(p.numel() for p in model.parameters())
