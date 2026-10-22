"""05-1b. 선택 결과의 통계 감사 (학습 없이) → select/<name>/audit.json

    python 05_sampling_audit.py --select random_r30
    python 05_sampling_audit.py --all                 # select/* 전부

모집단 = 학습 구간 정상 전체, 표본 = selected.parquet 의 정상. 지표:
  feature_kl_raw / feature_kl_ipw : 17개 엣지 피처별 KL(모집단 ‖ 표본)의 평균·최대 (모집단 분위수 20구간 히스토그램, IPW 가중 버전도)
  feature_psi_raw, feature_ks_raw : 같은 구간의 PSI 평균, KS 통계 평균
  daily_keep_min/max/ratio        : 일별 (표본 정상 / 모집단 정상) 의 최소·최대·최대÷최소 (1 이면 날짜 편향 없음)
  hour_jsd                        : 시간대(0~23) 분포 JSD(모집단, 표본)
  hard_frac_pop / hard_frac_sel   : 세탁 계좌가 낀 정상 비율 (모집단 vs 표본) — 경계 보존
  sibling_frac_pop / _sel         : 세탁 거래와 같은 송신계좌·같은 분인 정상 비율
  struct_pos_hop1_keep            : 구조 샘플링을 가정했을 때, 양성 간선 양끝 계좌의 1홉 간선 중 남는 비율
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, pandas as pd
import sampling_lib as L

BINS = 20


def hist_divergences(pop: np.ndarray, sel: np.ndarray, w: np.ndarray | None = None) -> dict:
    """열별 KL·PSI·KS. 구간은 모집단 분위수. 빈 구간엔 1e-8."""
    kl, psi, ks = [], [], []
    for j in range(pop.shape[1]):
        cuts = np.unique(np.quantile(pop[:, j], np.linspace(0, 1, BINS + 1)[1:-1])); edges = np.r_[-np.inf, cuts, np.inf]
        p = np.histogram(pop[:, j], edges)[0].astype(float) + 1e-8; p /= p.sum()
        q = np.histogram(sel[:, j], edges, weights=w)[0].astype(float) + 1e-8; q /= q.sum()
        kl.append(float(np.sum(p * np.log(p / q)))); psi.append(float(np.sum((p - q) * np.log(p / q))))
        ks.append(float(np.max(np.abs(np.cumsum(p) - np.cumsum(q)))))
    return dict(kl_mean=float(np.mean(kl)), kl_max=float(np.max(kl)), kl_max_feature=L.EDGE_FEATURES[int(np.argmax(kl))],
                psi_mean=float(np.mean(psi)), ks_mean=float(np.mean(ks)))


def jsd(p, q):
    p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2
    f = lambda a, b: float(np.sum(np.where(a > 0, a * np.log(a / b), 0)))
    return (f(p, m) + f(q, m)) / 2


def audit_one(name: str, con, store: L.GraphStore):
    d = L.SAMPLING / "select" / name; t0 = time.time()
    sel = pd.read_parquet(d / "selected.parquet"); meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    window = L.parse_window(meta["train"]); a, b = window
    lab = L.load_labels(con, window, patterns=False)
    pop = lab[lab.y == 0]; smp = sel[sel.y == 0]
    out = dict(select=name, method=meta["method"], ratio=meta["ratio"], n_pop=len(pop), n_sel=len(smp), keep_rate=len(smp) / len(pop))

    # 1) 피처 분포 — 모집단은 최대 200만 건 무작위 참조(메모리), 표본은 전부
    rng = np.random.default_rng(0)
    ref = pop.edge_id.to_numpy()[rng.choice(len(pop), min(len(pop), 2_000_000), replace=False)]
    X_pop = np.asarray(store.edge_x[np.sort(ref)]); X_sel = np.asarray(store.edge_x[smp.edge_id.to_numpy()])
    out["feature_raw"] = hist_divergences(X_pop, X_sel)
    out["feature_ipw"] = hist_divergences(X_pop, X_sel, smp.ipw.to_numpy())

    # 2) 시간 구조
    kp = smp.groupby("day").size() / pop.groupby("day").size()
    out.update(daily_keep_min=float(kp.min()), daily_keep_max=float(kp.max()), daily_keep_ratio=float(kp.max() / max(kp.min(), 1e-12)))
    tx = L.q(L.BASE / "transactions.parquet")
    con.register("sel_ids", smp[["edge_id"]])
    hp = con.execute(f"SELECT hour(ts) h, count(*) n FROM read_parquet({tx}) WHERE day BETWEEN DATE '{a}' AND DATE '{b}' AND NOT is_laundering GROUP BY 1 ORDER BY 1").df()
    hs = con.execute(f"SELECT hour(t.ts) h, count(*) n FROM read_parquet({tx}) t JOIN sel_ids USING (edge_id) GROUP BY 1 ORDER BY 1").df()
    out["hour_jsd"] = jsd(hp.set_index("h").n.reindex(range(24), fill_value=0).to_numpy(float), hs.set_index("h").n.reindex(range(24), fill_value=0).to_numpy(float))

    # 3) 경계 보존 — 세탁 계좌 연루 / 같은 분 sibling
    con.execute(f"""CREATE OR REPLACE TEMP TABLE t AS SELECT edge_id, ts, src, dst, is_laundering FROM read_parquet({tx})
                    WHERE day BETWEEN DATE '{a}' AND DATE '{b}'""")
    con.execute("CREATE OR REPLACE TEMP TABLE pos_nodes AS SELECT src AS node_id FROM t WHERE is_laundering UNION SELECT dst FROM t WHERE is_laundering")
    con.execute("""CREATE OR REPLACE TEMP TABLE flags AS
        SELECT n.edge_id, (n.src IN (SELECT node_id FROM pos_nodes) OR n.dst IN (SELECT node_id FROM pos_nodes)) AS hard,
               EXISTS (SELECT 1 FROM t p WHERE p.is_laundering AND p.src = n.src AND p.ts = n.ts AND p.edge_id <> n.edge_id) AS sibling
        FROM t n WHERE NOT n.is_laundering""")
    fp = con.execute("SELECT avg(hard::INT), avg(sibling::INT) FROM flags").fetchone()
    fs = con.execute("SELECT avg(hard::INT), avg(sibling::INT) FROM flags JOIN sel_ids USING (edge_id)").fetchone()
    out.update(hard_frac_pop=float(fp[0]), hard_frac_sel=float(fs[0]), sibling_frac_pop=float(fp[1]), sibling_frac_sel=float(fs[1]))

    # 4) 구조 샘플링을 했을 때 양성 1홉 보존율 (adjacency 기준이 아니라 전체 간선 기준의 근사)
    row = con.execute("""SELECT count(*) FILTER (WHERE s.edge_id IS NOT NULL OR n.is_laundering)::DOUBLE / count(*)
                         FROM t n LEFT JOIN sel_ids s USING (edge_id)
                         WHERE n.src IN (SELECT node_id FROM pos_nodes) OR n.dst IN (SELECT node_id FROM pos_nodes)""").fetchone()
    out["struct_pos_hop1_keep"] = float(row[0])
    out["seconds"] = round(time.time() - t0, 1)
    L.json_write(d / "audit.json", out)
    L.log(f"{name}: KL raw {out['feature_raw']['kl_mean']:.5f} / ipw {out['feature_ipw']['kl_mean']:.5f}, 일별보존 {out['daily_keep_ratio']:.2f}배, "
          f"hard {out['hard_frac_pop']:.3f}→{out['hard_frac_sel']:.3f}, 양성1홉보존 {out['struct_pos_hop1_keep']:.3f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--select"); ap.add_argument("--all", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--features", default="features"); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--memory", default="20GB")
    a = ap.parse_args(argv)
    names = sorted(p.name for p in (L.SAMPLING / "select").iterdir() if (p / "selected.parquet").exists()) if a.all else [a.select]
    con = L.connect(a.threads, a.memory); store = L.GraphStore(a.features, threads=a.threads, memory=a.memory); store.prepare(); store.load()
    for n in names:
        if (L.SAMPLING / "select" / n / "audit.json").exists() and not a.force:
            L.log(f"감사 있음: {n}"); continue
        audit_one(n, con, store)


if __name__ == "__main__":
    main()
