"""05-3. runs/*/run.json → 비교표·그림.   python 05_sampling_collect.py"""
import json, glob
import pandas as pd, numpy as np
import sampling_lib as L


def main(show=True):

    audits = {}
    for p in glob.glob(str(L.SAMPLING / "select" / "*" / "audit.json")):
        x = json.loads(open(p, encoding="utf-8").read())
        audits[x["select"]] = dict(kl_raw=x["feature_raw"]["kl_mean"], kl_ipw=x["feature_ipw"]["kl_mean"], psi_raw=x["feature_raw"]["psi_mean"],
                                   daily_keep_ratio=x["daily_keep_ratio"], hour_jsd=x["hour_jsd"], hard_frac_sel=x["hard_frac_sel"],
                                   hard_frac_pop=x["hard_frac_pop"], struct_pos_hop1_keep=x["struct_pos_hop1_keep"])
    if audits:
        A = pd.DataFrame(audits).T; A.index.name = "select"; A.to_csv(L.SAMPLING / "results" / "audit.csv") if (L.SAMPLING / "results").exists() else None
        print("=== 선택 결과 통계 감사 (학습 무관) ==="); print(A.round(5).to_string()); print()

    rows = []
    for p in glob.glob(str(L.SAMPLING / "runs" / "*" / "run.json")):
        r = json.loads(open(p, encoding="utf-8").read()); v = r.pop("val"); s = r.pop("sampling")
        rows.append({**r, **v, **{"keep_" + k[11:]: val for k, val in s.items()}, **audits.get(r["select"], {})})
    df = pd.DataFrame(rows)
    if df.empty:
        print("run.json 없음"); return None, None
    df["ratio"] = df.ratio.fillna(np.inf)        # none → ∞
    out = L.SAMPLING / "results"; out.mkdir(exist_ok=True)
    df.sort_values(["method", "apply", "reweight", "pi_eff", "ratio", "seed"]).to_csv(out / "runs.csv", index=False)

    keys = ["select", "method", "ratio", "apply", "reweight", "pi_eff"]
    metrics = ["AP", "F1", "recall_at_1pct", "unmatched_recall_at_1pct", "ECE", "train_sec_per_epoch"]
    summ = df.groupby(keys)[metrics].agg(["mean", "std", "count"]).round(4)
    summ.to_csv(out / "summary.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(summ[[("AP", "mean"), ("AP", "std"), ("AP", "count"), ("recall_at_1pct", "mean"), ("ECE", "mean"), ("train_sec_per_epoch", "mean")]])

    if "kl_raw" in df.columns and df.kl_raw.notna().sum() >= 4:
        from scipy.stats import spearmanr
        us = df[df.method != "none"].dropna(subset=["kl_raw"])
        rho, pv = spearmanr(us.kl_raw, us.AP); print(f"\n분포 왜곡(KL raw) ↔ AP 스피어만 상관 {rho:+.3f} (p={pv:.3f}, n={len(us)})")

    base = df[df.method == "none"]
    if len(base):
        print(f"\n샘플링 없음 기준선 AP {base.AP.mean():.4f} ± {base.AP.std():.4f} (n={len(base)})")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for (meth, ap_, rw, pi), g in df[df.method != "none"].assign(method=lambda d: d.select.str.replace(r"_r[0-9.]+", "", regex=True)).groupby(["method", "apply", "reweight", "pi_eff"]):
            m = g.groupby("ratio").AP.agg(["mean", "std"]).sort_index()
            for ax, col in zip(axes, ["AP", "recall_at_1pct", "ECE"]):
                mm = g.groupby("ratio")[col].agg(["mean", "std"]).sort_index()
                ax.errorbar(mm.index, mm["mean"], yerr=mm["std"].fillna(0), marker="o", capsize=3, label=f"{meth} {ap_} {rw} π={pi}")
        for ax, col in zip(axes, ["AP", "recall@1%", "ECE"]):
            if len(base): ax.axhline(base[col if col != "recall@1%" else "recall_at_1pct"].mean(), color="k", ls=":", label="none")
            ax.set_xscale("log"); ax.set_xlabel("neg:pos ratio r"); ax.set_title(col); ax.grid(alpha=.3)
        axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "ratio_curves.png", dpi=150)
        print(f"\n그림 → {out / 'ratio_curves.png'}")
        plt.show() if show else plt.close(fig)
    except Exception as e:
        print("그림 생략:", e)
    return df, summ


if __name__ == "__main__":
    main()
