"""05-1. 학습에 넣을 정상 거래 선택 → select/<method>_r<ratio>/selected.parquet

    python 05_sampling_select.py --method random --ratio 30 --train 2022-08-08:2022-09-27
    python 05_sampling_select.py --method hard   --ratio 30 --hard-frac 0.5
    python 05_sampling_select.py --method cssmc  --ratio 30 --clusters 30 --candidates 10
    python 05_sampling_select.py --method strata --ratio 30 --axes day,amount,cross_bank   # 다축 비례 층화
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


def select_strata(lab, ratio, con, window, axes, amount_bins, feat_dir, seed):
    """다축 비례 층화. 층 = axes 의 조합, 층별 할당 = 그날 예산 × (층 정상 수 / 그날 정상 수) 반올림.
    최소 1건 강제는 하지 않는다 (희소 층 과대표집 방지). stratum = 축값을 ':' 로 이은 문자열, IPW 층별.
      day        날짜
      amount     paid_currency_z (통화 보정 금액 z) 를 학습 정상 분위수로 amount_bins 구간
      recent     sender_count_1h (최근 1시간 송신 횟수, ln(1+n)) → 0 / 1~2 / 3~10 / 11+
      cross_bank from_bank <> to_bank
      new_pair   pair_prior_count == 0 (이 계좌쌍의 첫 거래)
    Reservoir 는 스트리밍용 구현이고 여기선 DuckDB 의 PARTITION BY + hash 순서가 같은 일을 한다."""
    a, b = window; tx = L.q(L.BASE / "transactions.parquet"); ef = L.q(feat_dir / "edge_features.parquet")
    b_day = budget_per_day(lab, ratio)
    con.register("budget", pd.DataFrame({"day": pd.to_datetime(b_day.index), "budget": b_day.to_numpy()}))
    con.execute(f"""CREATE OR REPLACE TEMP TABLE base AS
        SELECT t.edge_id, t.day, (t.from_bank <> t.to_bank) AS cross_bank, e.paid_currency_z, e.sender_count_1h, e.pair_prior_count
        FROM read_parquet({tx}) t JOIN read_parquet({ef}) e USING (edge_id)
        WHERE t.day BETWEEN DATE '{a}' AND DATE '{b}' AND NOT t.is_laundering""")
    cuts = con.execute(f"SELECT quantile_cont(paid_currency_z, [{', '.join(str(i / amount_bins) for i in range(1, amount_bins))}]) FROM base").fetchone()[0]
    expr = {"day": "day::VARCHAR",
            "amount": "CASE " + " ".join(f"WHEN paid_currency_z <= {c} THEN 'a{i}'" for i, c in enumerate(cuts)) + f" ELSE 'a{len(cuts)}' END",
            "recent": "CASE WHEN sender_count_1h = 0 THEN 'r0' WHEN sender_count_1h <= ln(3) THEN 'r1' WHEN sender_count_1h <= ln(11) THEN 'r2' ELSE 'r3' END",
            "cross_bank": "CASE WHEN cross_bank THEN 'x' ELSE 'i' END",
            "new_pair": "CASE WHEN pair_prior_count = 0 THEN 'new' ELSE 'old' END"}
    unknown = set(axes) - set(expr); assert not unknown, f"모르는 축 {unknown}"
    stratum = " || ':' || ".join(expr[x] for x in axes)
    picked = con.execute(f"""
        WITH s AS (SELECT edge_id, day, {stratum} AS stratum FROM base),
             n AS (SELECT stratum, day, count(*) AS n_s FROM s GROUP BY 1, 2),
             d AS (SELECT day, count(*) AS n_d FROM s GROUP BY 1),
             quota AS (SELECT n.stratum, n.day, n.n_s, least(n.n_s, round(b.budget * n.n_s / d.n_d)::BIGINT) AS k
                       FROM n JOIN d USING (day) JOIN budget b USING (day))
        SELECT s.edge_id, s.day, s.stratum, q.n_s::DOUBLE / q.k AS ipw
        FROM s JOIN quota q USING (stratum, day)
        QUALIFY row_number() OVER (PARTITION BY s.stratum, s.day ORDER BY hash(s.edge_id, {seed}), s.edge_id) <= q.k
        ORDER BY s.edge_id""").df()
    n_strata = con.execute("SELECT count(DISTINCT stratum) FROM (SELECT " + stratum + " AS stratum FROM base)").fetchone()[0]
    L.log(f"strata({','.join(axes)}): 층 {n_strata:,}개, 선택 {len(picked):,}건 (예산 합 {int(b_day.sum()):,})")
    con.execute("DROP TABLE base"); con.unregister("budget")
    return lab.merge(picked[["edge_id", "stratum", "ipw"]], on="edge_id", how="inner")


def select_cssmc(lab, ratio, rng, con, feat_dir, clusters, candidates, ref_rows=500_000, kl_rows=2_000_000, chunk=5_000_000):
    """CSSMC 축약판 (은총님 §6 와 세부가 다를 수 있음 — 비교 전에 맞출 것).
    1) 학습 정상 중 ref_rows 건 무작위 참조로 표준화 + MiniBatchKMeans
    2) 전체 학습 정상에 군집 라벨 — edge_features.parquet 를 edge_id 구간 청크로 읽어 predict (메모리: 청크 × 17 × 4B)
    3) 군집 비례 예산으로 후보 subset candidates 개 (군집 안 순서의 시작 오프셋만 다름)
    4) 후보마다 KL(모집단 ‖ 후보) — 모집단·후보 각각 kl_rows 표본, 피처는 parquet 1회 조인으로 읽음
    stratum = 군집, IPW = 군집 모집단 / 군집 선택 수."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    neg_ids = np.sort(lab.loc[lab.y == 0, "edge_id"].to_numpy(np.int64)); N = len(neg_ids)
    # 1) 참조 표본으로 군집 적합
    ref_ids = np.sort(rng.choice(neg_ids, min(N, ref_rows), replace=False))
    Xr = L.read_edge_features(con, feat_dir, ref_ids)[L.EDGE_FEATURES].to_numpy(np.float32)
    sc = StandardScaler().fit(Xr); km = MiniBatchKMeans(clusters, random_state=0, batch_size=8192, n_init=3).fit(sc.transform(Xr))
    # 2) 전체 정상 라벨 (청크)
    labels = np.empty(N, np.int16); con.register("neg_ids_all", pd.DataFrame({"edge_id": neg_ids}))
    cols = ", ".join(f'e."{c}"' for c in L.EDGE_FEATURES)
    for a in range(0, int(neg_ids[-1]) + 1, chunk):
        df = con.execute(f"""SELECT e.edge_id, {cols} FROM read_parquet({L.q(feat_dir / 'edge_features.parquet')}) e JOIN neg_ids_all USING (edge_id)
                             WHERE e.edge_id >= {a} AND e.edge_id < {a + chunk} ORDER BY e.edge_id""").df()
        if len(df):
            pos = np.searchsorted(neg_ids, df.edge_id.to_numpy(np.int64))
            labels[pos] = km.predict(sc.transform(df[L.EDGE_FEATURES].to_numpy(np.float32)))
        L.log(f"  cssmc 라벨 {min(a + chunk, int(neg_ids[-1]) + 1):,}/{int(neg_ids[-1]) + 1:,}")
    con.unregister("neg_ids_all")
    counts = np.bincount(labels, minlength=clusters)
    total = int(budget_per_day(lab, ratio).sum()); quota = np.minimum(counts, np.round(counts / counts.sum() * total)).astype(int)
    # 3) 후보 subset (군집별 멤버를 hash 순으로 섞은 뒤 오프셋)
    order = np.argsort(labels, kind="stable"); offs = np.searchsorted(labels[order], np.arange(clusters + 1))
    members = [order[offs[c]:offs[c + 1]] for c in range(clusters)]
    members = [m[rng.permutation(len(m))] for m in members]
    cands = []
    for cand in range(candidates):
        idx = np.concatenate([m[(np.arange(quota[c]) * candidates + cand) % len(m)] for c, m in enumerate(members) if len(m) and quota[c]])
        cands.append(np.unique(idx))
    # 4) KL — 모집단 표본 + 후보별 표본의 피처를 한 번에 읽음
    pop_s = np.sort(rng.choice(neg_ids, min(N, kl_rows), replace=False))
    cand_s = [np.sort(rng.choice(neg_ids[c], min(len(c), kl_rows), replace=False)) for c in cands]
    feats = L.read_edge_features(con, feat_dir, np.unique(np.concatenate([pop_s, *cand_s]))).set_index("edge_id")[L.EDGE_FEATURES]
    Xp = feats.loc[pop_s].to_numpy(np.float32)
    bins = [np.unique(np.quantile(Xp[:, j], np.linspace(0, 1, 21)[1:-1])) for j in range(Xp.shape[1])]
    P = [np.histogram(Xp[:, j], np.r_[-np.inf, bins[j], np.inf])[0] + 1e-8 for j in range(Xp.shape[1])]; P = [p / p.sum() for p in P]
    best = None
    for cand, ids_s in enumerate(cand_s):
        Xc = feats.loc[ids_s].to_numpy(np.float32); v = 0.0
        for j in range(Xc.shape[1]):
            qq = np.histogram(Xc[:, j], np.r_[-np.inf, bins[j], np.inf])[0] + 1e-8; qq /= qq.sum()
            v += float(np.sum(P[j] * np.log(P[j] / qq)))
        v /= Xc.shape[1]; L.log(f"  cssmc 후보 {cand}: {len(cands[cand]):,}건 KL={v:.5f}")
        if best is None or v < best[0]: best = (v, cand)
    idx = cands[best[1]]; lab_c = labels[idx]; chosen = np.bincount(lab_c, minlength=clusters)
    pick = pd.DataFrame({"edge_id": neg_ids[idx], "stratum": ["c" + str(c) for c in lab_c], "ipw": counts[lab_c] / np.maximum(chosen[lab_c], 1)})
    return lab.merge(pick, on="edge_id", how="inner")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["none", "random", "hard", "cssmc", "strata"])
    ap.add_argument("--ratio", type=float, default=None, help="정상:양성 (일별). none 이면 생략")
    ap.add_argument("--train", default="2022-08-08:2022-09-27")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--clusters", type=int, default=30); ap.add_argument("--candidates", type=int, default=10)
    ap.add_argument("--axes", default="day,amount,cross_bank", help="strata 축 (day,amount,recent,cross_bank,new_pair 중 콤마 구분)")
    ap.add_argument("--amount-bins", type=int, default=5)
    ap.add_argument("--features", default="features"); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--memory", default="12GB")
    a = ap.parse_args(argv)
    assert a.method == "none" or a.ratio, "--ratio 필요"
    name = "none" if a.method == "none" else f"{a.method}_r{a.ratio:g}"
    if a.method == "strata":
        name += "_" + "-".join(x.strip() for x in a.axes.split(","))
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
    elif a.method == "strata":
        neg = select_strata(lab, a.ratio, con, window, [x.strip() for x in a.axes.split(",")], a.amount_bins, L.features_dir(a.features), a.seed)
    else:
        neg = select_cssmc(lab, a.ratio, rng, con, L.features_dir(a.features), a.clusters, a.candidates)
    sel = pd.concat([pos, neg]).sort_values("edge_id").reset_index(drop=True)
    sel.to_parquet(out / "selected.parquet", index=False)
    daily = sel.groupby("day").agg(pos=("y", "sum"), neg=("y", lambda s: int((s == 0).sum())))
    L.json_write(out / "meta.json", dict(method=a.method, ratio=a.ratio, train=a.train, seed=a.seed, hard_frac=a.hard_frac,
                                         clusters=a.clusters, candidates=a.candidates, axes=(a.axes if a.method == "strata" else None), rows=len(sel), pos=int(pos.shape[0]), neg=int(neg.shape[0]),
                                         neg_population=int((lab.y == 0).sum()), ipw_sum_neg=float(neg.ipw.sum()),
                                         daily_keep_min=float((daily.neg / lab[lab.y == 0].groupby("day").size()).min()),
                                         daily_keep_max=float((daily.neg / lab[lab.y == 0].groupby("day").size()).max()),
                                         seconds=round(time.time() - t0, 1)))
    L.log(f"{name}: 양성 {len(pos):,} + 정상 {len(neg):,} (모집단 {(lab.y == 0).sum():,}) → {out}")


if __name__ == "__main__":
    main()
