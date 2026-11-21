"""Multi-FraudGT — 서비스(money_laundry dev) 이진 모델의 GNN 을 binary_classification 실험 입력에 맞춰 옮긴 것.

원본: money_laundry `backend/inference-server/models/multi_fraudgt_hgb/backend_model/models.py` (variant = multi_fraudgt_format)
      학습 방법은 docs 브랜치 `docs/02_data/taehoon/FraudGT_GNN_결과/fraudgt_lab/` (config.py · training.py · optimization.py) = FraudGT 공식 설정

그대로 가져온 것 (dev = 공식 FraudGT Large-HI Multi 설정)
  - FraudGTBlock 수식 (엣지 기반 attention bias, 메시지 gate, 엣지 임베딩 갱신) — 아래 클래스 본문은 원본과 같다
  - hidden 64 · head 8 · 2층 · attention dropout 0.2 · 분류 헤드 192 → 192 → 1
  - 학습 readout = official (이웃 샘플에 뽑힌 target 만 손실에 들어간다), 평가 readout = all_targets (모든 target 채점)
  - 손실 = 출력 1개 + BCE 양성 가중, AdamW (lr 0.001 · weight decay 1e-5), warmup 5 에폭 + cosine, grad clip 1.0, 8 배치 누적 후 1 step

실험에 맞춰 바꾼 것 (12_downsample_fraudgt.ipynb 머리말의 확정 설정)
  - 노드 입력: 계좌 피처 19 + ego → [상수 1, ego] 2열 (base 피처는 계좌 값을 엣지에 싣는다)
  - 엣지 입력: v10 14 + 결제방식 8 + 관계별 port 1 → base 20열 + port 2열 (양쪽 관계에 모두, rev_to 는 교환, z-norm) — 09 와 같은 행렬
  - 양성 가중 6 → 6.27, cosine 500 에폭 기준 → 250 에폭 기준, 체크포인트 선택 검증 F1@0.5 → 검증 AP, 검증 표본 = 09 고정 표본

배치 규약은 bc_lib 과 같다 (FGTStore 가 만드는 LinkNeighborLoader 배치). bc_lib 은 건드리지 않는다
"""
from __future__ import annotations
import gc, json, math, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import bc_lib as B

TO, REV = B.TO, B.REV


# ============================================================================ 설정
@dataclass
class FGTConfig:
    # 모델 (dev = 공식 설정)
    hidden: int = 64
    heads: int = 8
    layers: int = 2
    attention_dropout: float = 0.2
    # 손실 · 최적화
    pos_weight: float = 6.27          # BCE 양성 가중 (dev 는 6.0)
    base_lr: float = 0.001
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    max_epochs: int = 250             # 여기까지 조기 종료 없이 돈다
    schedule_epochs: int = 250        # cosine 이 0 에 닿는 에폭 (dev 는 500 기준으로 두고 284 에서 멈췄다)
    batch_accumulation: int = 8       # 로더 배치 8개의 평균 gradient 를 더한 뒤 1 step
    gradient_clip: float = 1.0
    train_readout: str = 'official'   # 'official' | 'all_targets'. 평가는 항상 all_targets
    # 구현 (결과에는 영향 없음)
    checkpoint_layers: bool = True    # 층마다 activation checkpointing — 역전파 때 다시 계산해 GPU 메모리를 줄인다 (dev 와 같음)

    def __post_init__(self):
        assert self.train_readout in ('official', 'all_targets')
        assert self.hidden % self.heads == 0 and self.max_epochs <= self.schedule_epochs

    def learning_rate(self, epoch):
        """fraudgt_lab.config.TrainConfig.learning_rate 와 같은 식. epoch 는 1 부터"""
        step = epoch - 1
        if step < self.warmup_epochs:
            factor = max(1e-6, step / self.warmup_epochs)
        else:
            factor = max(0., .5 * (1 + math.cos(math.pi * (step - self.warmup_epochs) / (self.schedule_epochs - self.warmup_epochs))))
        return self.base_lr * factor


# ============================================================================ 그래프 저장소 — bc_lib.GraphStore + target 행 번호
class WithTargetRows(B.AddEgoIdsAndTargetAttr):
    """bc_lib 배치 변환 + seed 간선의 tx 행 번호(batch.target_rows). official readout 이 '이 target 이 샘플된 간선 중 어디 있나' 를 찾는 데 쓴다"""
    def __call__(self, batch):
        batch = super().__call__(batch)
        batch.target_rows = self.target_rows[batch[TO].input_id].long()
        return batch


class FGTStore(B.GraphStore):
    """bc_lib.GraphStore 와 같고 배치 변환만 WithTargetRows 로 바꾼다 (make_loader 본문은 bc_lib 과 같다).
    이웃 샘플링은 PyG LinkNeighborLoader 기본값 = 균등 · 비복원 · directional (dev · 공식과 같음)"""

    def make_loader(self, graph, targets, shuffle):
        from torch_geometric.loader import LinkNeighborLoader
        cfg = self.cfg
        rev_neighbors = cfg.num_neighbors if cfg.use_reverse_mp else [0] * len(cfg.num_neighbors)
        return LinkNeighborLoader(
            graph, num_neighbors={TO: cfg.num_neighbors, REV: rev_neighbors},
            edge_label_index=(TO, graph[TO].edge_index[:, targets]), edge_label=graph[TO].y[targets],
            batch_size=cfg.batch_size if shuffle else cfg.eval_batch_size, shuffle=shuffle,
            transform=WithTargetRows(self.FEAT16, self.PORT_COLS, graph[TO].orig[targets], use_ego=cfg.use_ego, use_ports=cfg.use_ports),
            is_sorted=True, num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)

    def set_train_rows(self, rows):
        """학습 target 을 바꾼다 (loss 샘플링: 그래프 · 이웃은 그대로, 손실에 넣을 거래만 교체). 올라와 있는 그래프 · loader 는 버린다"""
        self.rows['train'] = np.asarray(rows, dtype=np.int64)
        self.current.update(name=None, graph=None, targets=None, loader=None); gc.collect()


# ============================================================================ 모델
def make_fraudgt(edge_dim, fcfg, node_dim=2):
    import torch
    from torch import nn
    from torch_geometric.utils import softmax

    class FraudGTBlock(nn.Module):
        """Sparse attention, attribute bias, gated values and updated edge embeddings. (dev models.py 의 FraudGTBlock 그대로)"""

        def __init__(self, relations: int, hidden: int = 64, heads: int = 8, attention_dropout: float = 0.2):
            super().__init__()
            self.hidden = hidden
            self.heads = heads
            self.node_norm = nn.LayerNorm(hidden)
            self.node_ffn_norm = nn.LayerNorm(hidden)
            self.edge_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(relations)])
            self.query = nn.Linear(hidden, hidden)
            self.key = nn.Linear(hidden, hidden)
            self.value = nn.Linear(hidden, hidden)
            self.node_output = nn.Linear(hidden, hidden)
            self.edge_bias = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(relations)])
            self.edge_gate = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(relations)])
            self.edge_output = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(relations)])
            self.relation_keys = nn.Parameter(torch.empty(relations, heads, hidden // heads, hidden // heads))
            nn.init.xavier_uniform_(self.relation_keys)
            self.node_ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
            self.edge_ffn = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)) for _ in range(relations)])
            self.attention_dropout = nn.Dropout(attention_dropout)

        def forward(self, nodes: torch.Tensor, indices: list, edges: list):
            normalized = self.node_norm(nodes)
            shape = (-1, self.heads, self.hidden // self.heads)
            query = self.query(normalized).view(shape)
            key = self.key(normalized).view(shape)
            value = self.value(normalized).view(shape)
            destinations, scores, messages, next_edges = [], [], [], []
            for relation, (index, attributes) in enumerate(zip(indices, edges)):
                source, destination = index
                edge_normalized = self.edge_norm[relation](attributes)
                # Transform each node once per relation, instead of allocating an E×H×D×D tensor.
                transformed_key = torch.einsum("hij,nhj->nhi", self.relation_keys[relation], key)
                vector_score = query[destination] * transformed_key[source] + self.edge_bias[relation](edge_normalized).view(shape)
                scalar_score = (vector_score.sum(-1) / math.sqrt(self.hidden // self.heads)).clamp(-5, 5)
                message = value[source] * self.edge_gate[relation](edge_normalized).view(shape).sigmoid()
                updated = attributes + self.edge_output[relation](vector_score.reshape(-1, self.hidden))
                updated = updated + self.edge_ffn[relation](updated)
                destinations.append(destination)
                scores.append(scalar_score)
                messages.append(message)
                next_edges.append(updated)
            destination = torch.cat(destinations)
            attention = softmax(torch.cat(scores), destination, num_nodes=len(nodes))
            message = self.attention_dropout(attention).unsqueeze(-1) * torch.cat(messages)
            aggregate = nodes.new_zeros((len(nodes), self.heads, self.hidden // self.heads))
            aggregate.index_add_(0, destination, message)
            updated_nodes = nodes + self.node_output(aggregate.flatten(1))
            # Preserve the published code's normalization/FFN ordering exactly.
            normalized_nodes = self.node_ffn_norm(updated_nodes)
            updated_nodes = normalized_nodes + self.node_ffn(normalized_nodes)
            return updated_nodes, next_edges

        def forward_queries(self, nodes, indices, edges, target_index, target_edges):
            """Read every seed without adding unsampled edges to message passing.

            Use the same relation-0 edge update as the published block. This repairs
            the upstream readout's omission of seeds absent from sampled e_id.
            No new parameters, neighbors, attention edges or labels are introduced.
            """
            updated_nodes, next_edges = self.forward(nodes, indices, edges)
            normalized = self.node_norm(nodes)
            shape = (-1, self.heads, self.hidden // self.heads)
            query = self.query(normalized).view(shape)
            key = self.key(normalized).view(shape)
            transformed = torch.einsum('hij,nhj->nhi', self.relation_keys[0], key)
            source, destination = target_index
            edge_normalized = self.edge_norm[0](target_edges)
            vector = query[destination] * transformed[source] + self.edge_bias[0](edge_normalized).view(shape)
            target_edges = target_edges + self.edge_output[0](vector.reshape(-1, self.hidden))
            target_edges = target_edges + self.edge_ffn[0](target_edges)
            return updated_nodes, next_edges, target_edges

    class MultiFraudGT(nn.Module):
        """dev FraudGT('multi_fraudgt_format') 와 같은 구성 — 관계 2개(to, rev_to), ego ID, port. 입력 차원만 실험 피처에 맞춘다.
        indices · edge_attrs 는 [to, rev_to] 순서. rev_to 의 edge_index 는 (수신 → 송신) 방향 그대로 준다 (dev 가 index.flip(0) 으로 만드는 것과 같은 방향)"""

        def __init__(self):
            super().__init__()
            h = fcfg.hidden
            self.node_encoder = nn.Linear(node_dim, h)                                    # dev: Linear(20, 64) (계좌 19 + ego)
            self.edge_encoder = nn.ModuleList([nn.Linear(edge_dim, h) for _ in range(2)])  # dev: Linear(22 + 1, 64) × 2
            self.layers = nn.ModuleList([FraudGTBlock(2, h, fcfg.heads, fcfg.attention_dropout) for _ in range(fcfg.layers)])
            self.head = nn.Sequential(nn.Linear(3 * h, 3 * h), nn.ReLU(), nn.Linear(3 * h, 1))
            self.checkpoint_layers = fcfg.checkpoint_layers

        def _run(self, fn, *args):
            if self.training and self.checkpoint_layers:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True)
            return fn(*args)

        def forward(self, x, indices, edge_attrs, target_index, target_attr, target_pos, readout):
            """→ (logits, selection). selection = 점수가 나온 target 의 배치 안 번호
            official    : target 간선이 샘플된 to 간선 안에 있을 때만 (target_pos >= 0) 그 간선의 갱신된 임베딩으로 채점
            all_targets : 모든 target. 빠진 target 에도 같은 관계 0 갱신식을 적용 (메시지 전달 간선은 늘리지 않는다)"""
            nodes = self.node_encoder(x)
            edges = [encoder(attributes) for encoder, attributes in zip(self.edge_encoder, edge_attrs)]
            if readout == 'official':
                for layer in self.layers:
                    nodes, edges = self._run(layer, nodes, indices, edges)
                selection = torch.nonzero(target_pos >= 0).flatten()
                source, destination = target_index[:, selection]
                representation = torch.cat([nodes[source], nodes[destination], edges[0][target_pos[selection]]], dim=1)
            elif readout == 'all_targets':
                queries = self.edge_encoder[0](target_attr)
                for layer in self.layers:
                    nodes, edges, queries = self._run(layer.forward_queries, nodes, indices, edges, target_index, queries)
                source, destination = target_index
                representation = torch.cat([nodes[source], nodes[destination], queries], dim=1)
                selection = torch.arange(len(source), device=nodes.device)
            else:
                raise ValueError(readout)
            return self.head(representation).squeeze(-1), selection

    return MultiFraudGT()


def target_positions(orig, target_rows):
    """target 마다 샘플된 to 간선 중 자기 자신의 위치 (없으면 -1). orig = 샘플된 to 간선들의 tx 행 번호, target_rows = seed 들의 tx 행 번호"""
    import torch
    pos = torch.full((len(target_rows),), -1, dtype=torch.long, device=target_rows.device)
    if len(orig) == 0 or len(target_rows) == 0:
        return pos
    sorted_orig, order = torch.sort(orig)
    k = torch.searchsorted(sorted_orig, target_rows).clamp_(max=len(orig) - 1)
    hit = sorted_orig[k] == target_rows
    pos[hit] = order[k[hit]]
    return pos


def forward_batch(model, batch, device, readout):
    """→ (logits, labels(float), 요청한 target 수). official 이면 logits 는 채점된 target 만"""
    batch = batch.to(device)
    pos = target_positions(batch[TO].orig.long(), batch.target_rows)
    logits, selection = model(batch['node'].x, [batch[TO].edge_index, batch[REV].edge_index],
                              [batch[TO].edge_attr, batch[REV].edge_attr], batch[TO].edge_label_index, batch.target_attr, pos, readout)
    return logits, batch[TO].edge_label[selection].float(), int(batch[TO].edge_label.numel())


def weighted_bce(logits, labels, pos_weight):
    """fraudgt_lab.optimization.class_weighted_loss — 양성에 pos_weight, 정상에 1. 평균 내기 전의 target 별 손실"""
    import torch
    import torch.nn.functional as F
    weights = torch.where(labels > 0, pos_weight, 1.0)
    return F.binary_cross_entropy_with_logits(logits, labels, weight=weights, reduction='none')


def make_optimizer(model, fcfg):
    """fraudgt_lab.optimization.make_optimizer 그대로 (이름에 bias 가 든 파라미터는 weight decay 제외)"""
    import torch
    named = list(model.named_parameters())
    no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight')
    groups = [
        {'params': [p for name, p in named if not any(key in name for key in no_decay)], 'weight_decay': fcfg.weight_decay},
        {'params': [p for name, p in named if any(key in name for key in no_decay)], 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(groups, lr=fcfg.learning_rate(1), weight_decay=fcfg.weight_decay)


# ============================================================================ 추론
def predict(model, store, name, device):
    """name('val' | 'test') 의 모든 target 점수 (sigmoid). readout = all_targets. loader 는 섞지 않으므로 store.rows[name] 순서 그대로"""
    import torch
    model.eval(); out = []
    with torch.inference_mode():
        for batch in store.use(name):
            logits, _, n = forward_batch(model, batch, device, 'all_targets')
            assert len(logits) == n, 'all_targets 인데 빠진 target 이 있다'
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f'{name} 점수에 nan/inf')
            out.append(torch.sigmoid(logits).float().cpu().numpy())
    score = np.concatenate(out)
    assert len(score) == len(store.rows[name]) or store.cfg.smoke, f'{name} 점수 수 {len(score)} != target 수 {len(store.rows[name])}'
    return score


# ============================================================================ 학습
def _settings(cfg, fcfg):
    """이어 돌릴 때 설정이 바뀌지 않았는지 확인할 값"""
    return dict(fgt=asdict(fcfg), num_neighbors=list(cfg.num_neighbors), batch_size=cfg.batch_size, eval_batch_size=cfg.eval_batch_size,
                iter_per_epoch=cfg.iter_per_epoch, eval_period=cfg.eval_period, seed=cfg.seed,
                use_reverse_mp=cfg.use_reverse_mp, use_ports=cfg.use_ports, use_ego=cfg.use_ego)


def _record(hist, cfg, fcfg, model=None, device=None):
    import torch
    best = hist.loc[hist['AP'].idxmax()]
    return dict(val={k: float(best[k]) for k in B.VAL_KEYS}, test=None, best_epoch=int(best['epoch']), epochs_run=len(hist),
                train_sec=int(hist['epoch_sec'].sum() + hist['val_sec'].fillna(0).sum()), epoch_sec=float(hist['epoch_sec'].mean()),
                targets_seen=int(best['epoch']) * cfg.iter_per_epoch * cfg.batch_size, coverage=float(hist['coverage'].mean()),
                gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device is not None and device.type == 'cuda' else None,
                kind='multi_fraudgt', n_params=sum(p.numel() for p in model.parameters()) if model is not None else None)


def train_fraudgt(name, store, cfg, fcfg, val_df, out_dir, device, results=None):
    """조건 하나 학습. store.rows['train'] 이 학습 target (set_train_rows 로 미리 넣는다)
    - 에폭 = cfg.iter_per_epoch 배치. 배치마다 (채점된 target 의 평균 손실).backward() → batch_accumulation 배치마다 clip · step
    - cfg.eval_period 에폭마다와 마지막 에폭에 검증 (고정 표본, all_targets). 검증 AP 최고 → <name>.pt
    - 조기 종료 없음. 에폭마다 <name>_last.pt · <name>_history.csv 저장 → 끊기면 같은 셀을 다시 돌려 그 에폭부터 이어 간다
    - 끝나면 <name>_done.json"""
    import torch
    from tqdm.auto import tqdm
    out_dir = Path(out_dir); best_path, last_path = out_dir / f'{name}.pt', out_dir / f'{name}_last.pt'
    B.seed_all(cfg.seed)
    model = make_fraudgt(store.edge_dim, fcfg).to(device); opt = make_optimizer(model, fcfg)
    settings = _settings(cfg, fcfg)
    history, best_ap, start = [], -np.inf, 1
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        assert state['settings'] == settings, f'{last_path.name} 의 설정이 지금과 다르다 — 이어 돌릴 수 없다. 새로 하려면 {name}* 파일을 지운다'
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        history, best_ap, start = state['history'], state['best_ap'], state['epoch']
        B.log(f'[{name}] {start} 에폭부터 이어서 (지금까지 최고 검증 AP {best_ap:.4f})')
    stream = None
    for epoch in range(start, fcfg.max_epochs + 1):
        lr = fcfg.learning_rate(epoch)
        for group in opt.param_groups:
            group['lr'] = lr
        if store.current['name'] != 'train' or stream is None:
            stream = B.InfiniteLoader(store.use('train'))
        model.train(); opt.zero_grad(set_to_none=True)
        loss_sum, scored, requested, te = 0.0, 0, 0, time.time()
        for it in tqdm(range(1, cfg.iter_per_epoch + 1), desc=f'{name} epoch {epoch}', leave=False):
            logits, labels, n = forward_batch(model, stream.next(), device, fcfg.train_readout)
            requested += n
            if len(logits):
                losses = weighted_bce(logits, labels, fcfg.pos_weight)
                if not torch.isfinite(losses).all():
                    raise FloatingPointError(f'[{name}] epoch {epoch} 학습 손실에 nan/inf')
                (losses.sum() / len(logits)).backward()                  # 배치 평균. zero_grad 를 안 했으므로 앞 배치의 gradient 에 더해진다
                loss_sum += float(losses.detach().sum()); scored += len(logits)
            if it % fcfg.batch_accumulation == 0 or it == cfg.iter_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), fcfg.gradient_clip)
                opt.step(); opt.zero_grad(set_to_none=True)
        row = dict(epoch=epoch, lr=lr, train_loss=loss_sum / max(scored, 1), coverage=scored / max(requested, 1), epoch_sec=round(time.time() - te), val_sec=None)
        if epoch % cfg.eval_period == 0 or epoch == fcfg.max_epochs:
            stream = None; tv = time.time()
            m = B.evaluate_sample(val_df, predict(model, store, 'val', device)); row.update(m); row['val_sec'] = round(time.time() - tv)
            print(f"epoch {epoch:3d}  lr {lr:.2e}  loss {row['train_loss']:.4f}  cov {row['coverage']:.2f}  AP {m['AP']:.4f}  F1 {m['F1']:.4f}  "
                  f"recall@1% {m['recall_at_1pct']:.4f}  (학습 {row['epoch_sec']}s + 검증 {row['val_sec']}s)")
            if m['AP'] > best_ap:
                best_ap = float(m['AP']); torch.save(model.state_dict(), best_path)
        else:
            print(f"epoch {epoch:3d}  lr {lr:.2e}  loss {row['train_loss']:.4f}  cov {row['coverage']:.2f}  ({row['epoch_sec']}s)")
        history.append({k: (None if v is None else float(v)) for k, v in row.items()})
        pd.DataFrame(history).to_csv(out_dir / f'{name}_history.csv', index=False)
        tmp = last_path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch + 1, history=history, best_ap=best_ap, settings=settings), tmp)
        tmp.replace(last_path)
    hist = pd.DataFrame(history)
    rec = _record(hist, cfg, fcfg, model, device)
    (out_dir / f'{name}_done.json').write_text(json.dumps(dict(settings=settings, **rec), ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    if results is not None:
        results[name] = rec
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, hist


def test_fraudgt(name, store, cfg, fcfg, test_df, out_dir, device, results):
    """검증 AP 최고 체크포인트로 테스트 구간 전체 추론 (all_targets) → <name>_test_scores.parquet"""
    import torch
    out_dir = Path(out_dir)
    assert (out_dir / f'{name}_done.json').exists(), f'{name} 학습이 끝나지 않았다'
    if name not in results:                     # 학습 셀을 건너뛰고 왔으면 done.json 에서 복원
        rec = json.loads((out_dir / f'{name}_done.json').read_text(encoding='utf-8')); rec.pop('settings', None); rec['test'] = None
        results[name] = rec
    model = make_fraudgt(store.edge_dim, fcfg).to(device)
    model.load_state_dict(torch.load(out_dir / f'{name}.pt', map_location=device, weights_only=True))
    t0 = time.time(); score = predict(model, store, 'test', device)
    pd.DataFrame(dict(edge_id=test_df.edge_id, y=test_df.y, score=score)).to_parquet(out_dir / f'{name}_test_scores.parquet', index=False)
    results[name]['test'] = B.evaluate_sample(test_df, score)
    print(name, f'({(time.time() - t0) / 60:.0f}분)', {k: round(v, 4) for k, v in results[name]['test'].items()})
    return score
