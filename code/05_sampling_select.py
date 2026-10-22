"""05-1. 학습에 넣을 정상 거래 선택 → select/<method>_r<ratio>/selected.parquet

    python 05_sampling_select.py --method random --ratio 30 --train 2022-08-08:2022-09-27
    python 05_sampling_select.py --method hard   --ratio 30 --hard-frac 0.5
    python 05_sampling_select.py --method cssmc  --ratio 30 --clusters 30 --candidates 10
    python 05_sampling_select.py --method none                      # 전부 (샘플링 없음 기준선)

출력 selected.parquet 컬럼
    edge_id, day, y(0/1), in_pattern, stratum(문자열), ipw(= 층 모집단 수 / 층 선택 수, 양성은 1)
양성(y=1)은 항상 전부 포함. ratio = 일별 (선택 정상 수) / (그날 양성 수). 적용 위치(지도/구조)는 여기서 정하지 않는다 — 같은 목록을 train 에서 둘 다 쓴다.
"""
from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd
import sampling_lib as L


def budget_per_day(lab: pd.DataFrame, ratio: float) -> pd.Series:
    """날짜별 정상 예산 = min(그날 정상 수, ceil(그날 양성 수 × ratio))."""
    g = lab.groupby("day").y.agg(pos="sum", n="size")
    want = np.ceil(g.pos * ratio)
    want[g.pos == 0] = np.ceil(g.pos[g.pos > 0].median() * ratio)     # 양성이 0인 날도 그날 정상을 비우지 않도록 중앙값 예산
    return np.minimum(g.n - g.pos, want).astype(int)


def select_random(lab, ratio, rng):
    """일별 층화 무작위. stratum = day."""
    b = budget_per_day(lab, ratio); parts = []
    for day, g in lab.groupby("day"):
        neg = g[g.y == 0]; k = int(b[day])
        pick = neg.iloc[rng.choice(len(neg), k, replace=False)] if k < len(neg) else neg
        parts.append(pick.assign(stratum=str(day), ipw=len(neg) / max(k, 1)))
    return pd.concat(parts)


def hard_pool(con, window) -> np.ndarray:
    """세탁 계좌가 낀 정상 거래 + 세탁 거래와 같은 송신계좌·같은 분의 정상 거래 (학습 구간 안)."""
    a, b = window; tx = L.q(L.BASE / "transactions.parquet")
    return con.execute(f"""
        WITH t AS (SELECT edge_id, ts, src, dst, is_laundering FROM read_parquet({tx}) WHERE day BETWEEN DATE '{a}' AND DATE '{b}'),
             pos_nodes AS (SELECT src AS node_id FROM t WHERE is_laundering UNION SELECT dst FROM t WHERE is_laundering)
        SELECT edge_id FROM t WHERE NOT is_laundering AND (src IN (SELECT node_id FROM pos_nodes) OR dst IN (SELECT node_id FROM pos_nodes))
        UNION
        SELECT n.edge_id FROM t p JOIN t n ON n.src = p.src AND n.ts = p.ts AND n.edge_id <> p.edge_id
        WHERE p.is_laundering AND NOT n.is_laundering""").df().edge_id.to_numpy()


def select_hard(lab, ratio, rng, con, window, hard_frac):
    """일별 예산의 hard_frac 은 hard 풀에서, 나머지는 그 외 정상에서. stratum = day:hard / day:easy (IPW 층별)."""
    hard_ids = hard_pool(con, window); L.log(f"hard 풀 {len(hard_ids):,}건")
    b = budget_per_day(lab, ratio); parts = []
    for day, g in lab.groupby("day"):
        neg = g[g.y == 0]; h = np.isin(neg.edge_id.to_numpy(), hard_ids)
        k_hard = int(min(h.sum(), round(b[day] * hard_frac))); k_easy = int(min((~h).sum(), b[day] - k_hard))
        k_hard = int(min(h.sum(), b[day] - k_easy))          # easy 풀이 모자라면 남는 예산을 hard 로
        for name, pool, k in [("hard", neg[h], k_hard), ("easy", neg[~h], k_easy)]:
            if k <= 0 or len(pool) == 0: continue
            pick = pool.iloc[rng.choice(len(pool), k, replace=False)] if k < len(pool) else pool
            parts.append(pick.assign(stratum=f"{day}:{name}", ipw=len(pool) / k))
    return pd.concat(parts)


def select_cssmc(lab, ratio, rng, store_feat: np.ndarray, clusters, candidates):
    """CSSMC 축약판 (은총님 §6 와 세부가 다를 수 있음 — 비교 전에 맞출 것).
    1) 학습 정상 표준화 17피처로 MiniBatchKMeans  2) 군집 비례 배분으로 후보 subset candidates 개 (시작 오프셋만 다름)
    3) 피처별 히스토그램 KL(모집단 ‖ 후보) 평균이 최소인 후보 선택.  stratum = cluster (IPW 군집별)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    neg = lab[lab.y == 0].reset_index(drop=True); X = store_feat[neg.edge_id.to_numpy()]
    ref = X[rng.choice(len(X), min(len(X), 500_000), replace=False)]
    sc = StandardScaler().fit(ref); km = MiniBatchKMeans(clusters, random_state=0, batch_size=8192, n_init=3).fit(sc.transform(ref))
    lab_c = km.predict(sc.transform(X)); counts = np.bincount(lab_c, minlength=clusters)
    total = int(budget_per_day(lab, ratio).sum()); quota = np.maximum(1, np.round(counts / counts.sum() * total)).astype(int)
    bins = [np.unique(np.quantile(ref[:, j], np.linspace(0, 1, 21)[1:-1])) for j in range(X.shape[1])]
    def kl(idx):
        v = 0.0
        for j in range(X.shape[1]):
            p = np.histogram(X[:, j], np.r_[-np.inf, bins[j], np.inf])[0] + 1e-8; p /= p.sum()
            qq = np.histogram(X[idx, j], np.r_[-np.inf, bins[j], np.inf])[0] + 1e-8; qq /= qq.sum()
            v += float(np.sum(p * np.log(p / qq)))
        return v / X.shape[1]
    order = [np.flatnonzero(lab_c == c) for c in range(clusters)]
    best = None
    for cand in range(candidates):
        idx = np.concatenate([o[(np.arange(min(quota[c], len(o))) * candidates + cand) % len(o)] for c, o in enumerate(order) if len(o)])
        idx = np.unique(idx); v = kl(idx)
        L.log(f"  cssmc 후보 {cand}: {len(idx):,}건 KL={v:.5f}")
        if best is None or v < best[0]: best = (v, idx)
    pick = neg.iloc[best[1]].copy(); pick["stratum"] = ["c" + str(c) for c in lab_c[best[1]]]
    chosen = np.bincount(lab_c[best[1]], minlength=clusters); pick["ipw"] = counts[lab_c[best[1]]] / np.maximum(chosen[lab_c[best[1]]], 1)
    return pick


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["none", "random", "hard", "cssmc"])
    ap.add_argument("--ratio", type=float, default=None, help="정상:양성 (일별). none 이면 생략")
    ap.add_argument("--train", default="2022-08-08:2022-09-27")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--clusters", type=int, default=30); ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--features", default="features"); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--memory", default="12GB")
    a = ap.parse_args(argv)
    assert a.method == "none" or a.ratio, "--ratio 필요"
    name = "none" if a.method == "none" else f"{a.method}_r{a.ratio:g}"
    out = L.SAMPLING / "select" / name; out.mkdir(parents=True, exist_ok=True)
    if (out / "selected.parquet").exists():
        L.log(f"이미 있음: {out}"); return
    con = L.connect(a.threads, a.memory); window = L.parse_window(a.train); rng = np.random.default_rng(a.seed); t0 = time.time()
    lab = L.load_labels(con, window)
    L.log(f"학습 구간 {window}: {len(lab):,}건, 양성 {int(lab.y.sum()):,}")
    pos = lab[lab.y == 1].assign(stratum="pos", ipw=1.0)
    if a.method == "none":
        neg = lab[lab.y == 0].assign(stratum="all", ipw=1.0)
    elif a.method == "random":
        neg = select_random(lab, a.ratio, rng)
    elif a.method == "hard":
        neg = select_hard(lab, a.ratio, rng, con, window, a.hard_frac)
    else:
        store = L.GraphStore(a.features, threads=a.threads, memory=a.memory); store.prepare(); store.load()
        neg = select_cssmc(lab, a.ratio, rng, store.edge_x, a.clusters, a.candidates)
    sel = pd.concat([pos, neg]).sort_values("edge_id").reset_index(drop=True)
    sel.to_parquet(out / "selected.parquet", index=False)
    daily = sel.groupby("day").agg(pos=("y", "sum"), neg=("y", lambda s: int((s == 0).sum())))
    L.json_write(out / "meta.json", dict(method=a.method, ratio=a.ratio, train=a.train, seed=a.seed, hard_frac=a.hard_frac,
                                         clusters=a.clusters, candidates=a.candidates, rows=len(sel), pos=int(pos.shape[0]), neg=int(neg.shape[0]),
                                         neg_population=int((lab.y == 0).sum()), ipw_sum_neg=float(neg.ipw.sum()),
                                         daily_keep_min=float((daily.neg / lab[lab.y == 0].groupby("day").size()).min()),
                                         daily_keep_max=float((daily.neg / lab[lab.y == 0].groupby("day").size()).max()),
                                         seconds=round(time.time() - t0, 1)))
    L.log(f"{name}: 양성 {len(pos):,} + 정상 {len(neg):,} (모집단 {(lab.y == 0).sum():,}) → {out}")


if __name__ == "__main__":
    main()
