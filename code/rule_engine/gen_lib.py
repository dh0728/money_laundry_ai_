"""Alert 생성기 실험 공통 도구.

서비스 저장소(money_laundry, dev 기준)의 worker/alerts 코드를 **그대로 불러와** 실험 데이터에 돌린다.
생성기 코드를 여기로 복사하지 않는다. 서비스 쪽이 바뀌면 이 실험 결과도 같이 바뀐다.

서비스와 같은 것
  - 씨앗 보강: alerts/seed_promotion.promote (low, floor 는 이진 모델 spec 의 값)
  - 묶기: alerts/block_graph.build_block_graphs + BlockPolicy() 기본값
  - 금액: 보낸 금액(paid)을 고정 환율(core.fx_rates, fx_rates_usd_v1)로 USD 환산
서비스와 다른 것
  - 서비스는 매일 60일 창을 읽고 이미 공개된 Alert 를 고정(pin)한 채 이어 붙인다.
    여기서는 평가 구간의 씨앗을 한 번에 넣고 한 번 돌린다. 날짜별 누적 결과와 다를 수 있다.
  - 거래 번호는 창 안에서 1..N 으로 다시 매긴다(local id = 행 번호 + 1). edge_id 와는 행 순서로 대응한다.
"""
from array import array
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import pandas as pd

# 서비스의 고정 환율: 1 USD 가 그 통화로 몇 단위인가.
# 출처: money_laundry backend/api-server/src/main/resources/db/migration/V1__initial_schema.sql (fx_rates_usd_v1)
UNITS_PER_USD = {
    'Australian Dollar': 1.41280000, 'Bitcoin': 0.00008416, 'Brazil Real': 5.64649997, 'Canadian Dollar': 1.31930001,
    'Euro': 0.85340000, 'Mexican Peso': 21.14310180, 'Ruble': 77.80399773, 'Rupee': 73.44397914,
    'Saudi Riyal': 3.75109996, 'Shekel': 3.37699997, 'Swiss Franc': 0.91500000, 'UK Pound': 0.77420000,
    'US Dollar': 1.00000000, 'Yen': 105.39995594, 'Yuan': 6.69760020,
}
ROLE_NONE, ROLE_SEED, ROLE_CONNECTION = 0, 1, 2   # 생성기 결과에서 거래가 맡은 역할


# ----------------------------------------------------------------------------- 생성기 불러오기
def find_worker(start=None):
    """나란히 있는 money_laundry 저장소에서 worker 폴더를 찾는다. dev 기준 구조(alerts/block_graph.py)여야 한다."""
    here = Path(start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        worker = p / 'money_laundry' / 'backend' / 'api-server' / 'worker'
        if (worker / 'alerts' / 'block_graph.py').is_file():
            return worker
        if worker.is_dir():
            raise FileNotFoundError(f'{worker} 에 alerts/block_graph.py 가 없다. money_laundry 를 dev 브랜치로 맞춰야 한다')
    raise FileNotFoundError('money_laundry 저장소를 못 찾음 (이 저장소와 나란히 있어야 한다)')


def load_generator(worker):
    """서비스 생성기 모듈을 불러온다. 표준 라이브러리만 쓰는 모듈이라 DB·서버 없이 돈다."""
    worker = str(worker)
    if worker not in sys.path:
        sys.path.insert(0, worker)
    from alerts.block_graph import BlockPolicy, build_block_graphs
    from alerts.flow_graph import EdgeTable, FlowIndex, GraphBudgetExceeded
    from alerts.seed_promotion import promote
    return SimpleNamespace(BlockPolicy=BlockPolicy, build_block_graphs=build_block_graphs, EdgeTable=EdgeTable,
                           FlowIndex=FlowIndex, GraphBudgetExceeded=GraphBudgetExceeded, promote=promote)


# ----------------------------------------------------------------------------- 입력 만들기
def to_usd(paid, currency):
    """서비스 통합 단계와 같은 환산: amount_usd = paid ÷ units_per_usd (소수 6자리). 모르는 통화는 NaN"""
    rate = pd.Series(currency).map(UNITS_PER_USD).to_numpy(dtype='float64')
    return np.round(np.asarray(paid, dtype='float64') / rate, 6)


def make_edges(G, tx):
    """(ts, edge_id) 순으로 정렬된 DataFrame[ts, src, dst, usd] → 생성기 입력(EdgeTable).

    local id 는 행 번호 + 1 이다. append 를 행마다 부르면 느려서 열을 통째로 넣는다(검사 항목은 아래 assert 로 대신한다).
    """
    n = len(tx)
    usd = tx['usd'].to_numpy(dtype='float64')
    columns = dict(
        tx_ids=np.arange(1, n + 1, dtype='<i8'),
        occurred_us=tx['ts'].to_numpy().astype('datetime64[us]').astype('<i8'),
        from_accounts=tx['src'].to_numpy().astype('<i8') + 1,     # 생성기는 계좌 번호가 1 이상이어야 한다
        to_accounts=tx['dst'].to_numpy().astype('<i8') + 1,
        amount_usd_micros=np.where(np.isfinite(usd), np.rint(usd * 1e6), -1).astype('<i8'))   # -1 = 금액 모름
    assert (np.diff(columns['occurred_us']) >= 0).all(), 'tx 는 시간순으로 정렬돼 있어야 한다'
    assert columns['from_accounts'].min() >= 1 and columns['to_accounts'].min() >= 1
    assert columns['amount_usd_micros'].min() >= -1
    edges = G.EdgeTable()
    for name, values in columns.items():
        column = array('q')
        column.frombytes(np.ascontiguousarray(values).tobytes())
        setattr(edges, name, column)
    return edges


# ----------------------------------------------------------------------------- 보강·묶기
def promote_seeds(G, edges, score, seed_rows, low, floor):
    """씨앗 보강(서비스 _promote_seeds 와 같은 호출). 새로 씨앗이 된 행 번호 배열을 돌려준다.

    score: 행마다의 모델 점수(없으면 NaN). seed_rows: 이미 씨앗인 행 번호.
    """
    lowest = min(low, floor)
    rows = np.flatnonzero(np.nan_to_num(score, nan=-1.0) >= lowest)
    probability = {int(r) + 1: float(score[r]) for r in rows}      # 점수가 lowest 이상인 거래만 본다(서비스도 그 아래는 안 본다)
    for r in seed_rows:                                            # 씨앗은 점수와 상관없이 들어 있어야 한다
        probability.setdefault(int(r) + 1, float(score[r]) if np.isfinite(score[r]) else 1.0)
    positions = G.promote(edges, probability, [int(r) + 1 for r in seed_rows], low, floor)
    return np.asarray(sorted(positions), dtype=np.int64)


def run_generator(G, edges, seed_rows, policy=None, budget_scale=1):
    """생성기 실행. budget_scale 은 탐색 예산(max_edge_visits, max_relations)만 키운다. 묶는 기준은 그대로다."""
    policy = policy or G.BlockPolicy()
    if budget_scale != 1:
        policy = replace(policy, max_edge_visits=policy.max_edge_visits * budget_scale,
                         max_relations=policy.max_relations * budget_scale)
    started = time.perf_counter()
    index = G.FlowIndex(edges)
    index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    batch = G.build_block_graphs(index, sorted(int(r) + 1 for r in seed_rows), policy)
    return batch, dict(policy=policy.version, index_seconds=round(index_seconds, 1),
                       build_seconds=round(time.perf_counter() - started, 1), edge_visits=batch.edge_visits,
                       graphs=len(batch.graphs), unassigned=len(batch.unassigned))


def batch_table(batch, n):
    """생성기 결과 → 행마다 (역할, Alert 번호, 탈락 사유) 와 Alert 마다의 요약.

    한 거래가 Alert 두 개에 들어가면 Alert 번호는 뒤의 것으로 남는다(겹친 수는 alerts['members'] 합으로 알 수 있다).
    """
    role = np.zeros(n, dtype=np.int8)
    graph = np.full(n, -1, dtype=np.int32)
    rows = []
    for g, block in enumerate(batch.graphs):
        seeds = np.asarray(block.seed_ids, dtype=np.int64) - 1
        links = np.asarray(block.connection_ids, dtype=np.int64) - 1
        role[links], graph[links] = ROLE_CONNECTION, g
        role[seeds], graph[seeds] = ROLE_SEED, g                   # 씨앗이 우선
        kinds = Counter(w.kind for w in block.witnesses)
        rows.append(dict(alert=g, seeds=len(seeds), connections=len(links), members=len(seeds) + len(links),
                         kinds=' '.join(sorted(kinds)), limits=' '.join(block.limits)))
    reason = np.full(n, '', dtype=object)
    for item in batch.unassigned:
        reason[item.tx_id - 1] = item.reason
    return role, graph, reason, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 지표
def stage_metrics(name, mask, y, eval_mask, pattern_mask):
    """평가 구간 안에서 mask 로 고른 거래의 정밀도·재현율. 재현율은 세탁 전체 기준과 패턴 세탁 기준 두 가지."""
    chosen = mask & eval_mask
    positive, patterned = eval_mask & (y == 1), eval_mask & pattern_mask
    tp, fp = int((chosen & (y == 1)).sum()), int((chosen & (y == 0)).sum())
    return dict(단계=name, 고른_거래=int(chosen.sum()), 세탁=tp, 정상=fp,
                정밀도=round(tp / max(tp + fp, 1), 4),
                재현율_전체=round(tp / max(int(positive.sum()), 1), 4),
                재현율_패턴만=round(int((chosen & patterned).sum()) / max(int(patterned.sum()), 1), 4))


def split_by_label(mask, y, pattern_type):
    """mask 로 고른 거래를 정상 / 세탁(패턴별) 로 센다. pattern_type: 패턴 없는 세탁은 'NO_PATTERN', 정상은 ''"""
    label = np.where(y == 1, pattern_type, '정상')
    return pd.Series(label[mask]).value_counts()


def alert_purity(graph, role, y, alerts):
    """Alert 마다 구성 거래(씨앗 + 연결) 중 세탁 비율."""
    member = role > ROLE_NONE
    frame = pd.DataFrame(dict(alert=graph[member], y=y[member]))
    purity = frame.groupby('alert')['y'].agg(laundering='sum', counted='size')
    out = alerts.merge(purity, left_on='alert', right_index=True, how='left')
    out['purity'] = out['laundering'] / out['counted']
    return out


# ----------------------------------------------------------------------------- 묶음을 정답(세탁 시도)과 비교 (02)
def link_by_account(seeds, hours=None):
    """단순 연결: 씨앗(DataFrame[ts, src, dst])을 '계좌를 공유하면 같은 묶음' 규칙으로 묶고 행마다 묶음 번호를 돌려준다.

    hours 를 주면 같은 계좌에서 시간순으로 이웃한 두 씨앗의 간격이 그 시간 이하일 때만 잇는다.
    금액, 방향, 바쁜 계좌는 보지 않는다. 생성기와 비교하기 위한 기준선이다.
    """
    n = len(seeds)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seconds = seeds['ts'].to_numpy().astype('datetime64[s]').astype('int64')
    ends = pd.DataFrame(dict(row=np.tile(np.arange(n), 2), t=np.tile(seconds, 2),
                             account=np.concatenate([seeds['src'].to_numpy(), seeds['dst'].to_numpy()])))
    ends = ends.drop_duplicates(['row', 'account']).sort_values(['account', 't', 'row'])
    account, row, t = ends['account'].to_numpy(), ends['row'].to_numpy(), ends['t'].to_numpy()
    linked = account[1:] == account[:-1]
    if hours is not None:
        linked &= (t[1:] - t[:-1]) <= hours * 3600
    for a, b in zip(row[:-1][linked], row[1:][linked]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def _pairs(n):
    return n * (n - 1) // 2


def compare_to_truth(name, groups, attempt_of, eval_attempts, truth=None):
    """묶음 하나(방법 하나)를 정답 시도와 비교한다.

    groups       : DataFrame[edge_id, group, y]. 묶음에 들어간 거래만(묶음 크기 2 이상). 한 거래는 한 묶음에만 있다
    attempt_of   : DataFrame[edge_id, attempt_id, pattern_type]. 패턴 세탁 거래 전부(평가 대상이 아닌 시도 포함)
    eval_attempts: 평가할 시도 번호들
    truth        : 정답으로 쓸 거래 DataFrame[edge_id, attempt_id, pattern_type]. 안 주면 eval_attempts 의 거래 전부
    돌려주는 것  : (요약 dict, 시도별 DataFrame)
    """
    g = groups.merge(attempt_of[['edge_id', 'attempt_id']], on='edge_id', how='left')
    size = g.groupby('group').size()
    # --- 묶음 쪽: 묶음이 얼마나 깨끗한가
    cell = g.dropna(subset=['attempt_id']).groupby(['group', 'attempt_id']).size()
    attempts_in = cell.groupby('group').size().reindex(size.index, fill_value=0)
    dominant = cell.groupby('group').max().reindex(size.index, fill_value=0)
    laundering = g.groupby('group')['y'].sum()
    same_pairs = int(sum(_pairs(int(v)) for v in cell.to_numpy()))
    all_pairs = int(sum(_pairs(int(v)) for v in size.to_numpy()))
    # --- 정답 쪽: 시도가 얼마나 온전히 한 묶음에 들어갔나
    if truth is None:
        truth = attempt_of[attempt_of['attempt_id'].isin(eval_attempts)]
    truth = truth[['edge_id', 'attempt_id', 'pattern_type']].merge(groups[['edge_id', 'group']], on='edge_id', how='left')
    rows = []
    for attempt, part in truth.groupby('attempt_id'):
        inside = part['group'].dropna()
        counts = inside.value_counts()
        rows.append(dict(attempt_id=attempt, pattern_type=part['pattern_type'].iloc[0], size=len(part),
                         covered=len(inside), best=int(counts.iloc[0]) if len(counts) else 0, n_groups=len(counts),
                         same_pairs=int(sum(_pairs(int(v)) for v in counts.to_numpy()))))
    per = pd.DataFrame(rows)
    per['coverage'] = per['best'] / per['size']
    summary = dict(
        방법=name, 묶음_수=len(size), 묶인_거래=int(size.sum()), 크기_중앙값=float(size.median()), 크기_최대=int(size.max()),
        세탁_비율=round(float(laundering.sum() / size.sum()), 4),
        전부_정상_묶음=int((laundering == 0).sum()),
        시도_2개이상_섞인_묶음=int((attempts_in >= 2).sum()),
        쌍_정밀도=round(same_pairs / max(all_pairs, 1), 4),          # 같은 묶음의 두 거래가 같은 시도일 비율
        쌍_재현율=round(per['same_pairs'].sum() / max(int(sum(_pairs(int(v)) for v in per['size'])), 1), 4),   # 같은 시도의 두 거래가 같은 묶음일 비율
        시도_수=len(per), 온전히_한_묶음=int((per['best'] == per['size']).sum()),
        온전_비율=round(float((per['best'] == per['size']).mean()), 4),
        평균_최대_포함률=round(float(per['coverage'].mean()), 4),
        쪼개진_시도=int((per['n_groups'] >= 2).sum()), 전혀_안_묶인_시도=int((per['covered'] == 0).sum()))
    return summary, per
