"""05-2. 실험 조건 1개 학습·평가 → runs/<run_name>/run.json

    python 05_sampling_train.py --select random_r30 --apply loss   --reweight none --pi-eff 0.25 --seed 42
    python 05_sampling_train.py --select random_r30 --apply struct --reweight none --pi-eff 0.25 --seed 42
    python 05_sampling_train.py --select none       --apply loss   --reweight none --pi-eff 0.25 --seed 42   # 샘플링 없음 기준선

실험 조건 = (select 목록) × (apply: loss | struct) × (reweight: ipw | none) × pi_eff × seed
  apply=loss   : 그래프는 전체, select 에 없는 학습 정상은 loss 에서만 빠짐 (지도 샘플링)
  apply=struct : select 에 없는 학습 정상을 adjacency 에서도 제거 (구조-A: 피처는 그대로).
                 피처까지 재계산한 구조-B 는 02 를 EDGE_FILTER 로 돌린 뒤 --features <그 폴더> 로.
  reweight=ipw : 선택된 정상에 IPW(모집단/선택) 가중치 → 전체 데이터 손실의 불편추정 (태훈님 방식)
  reweight=none: 가중치 1 → 모델이 보는 정상:양성 비율이 실제로 바뀜
  pi_eff       : 손실에서 양성 가중치 합의 비중. pos_weight 를 여기서 역산해 r 과 분리한다
                   pos_weight = pi/(1-pi) × Σ(정상 가중치) / 양성 수

검증: 검증 구간 양성 전부 + 정상의 --val-neg-rate 비율(일별 층화, 가중치 1/rate). 모든 조건이 같은 파일을 공유.
      (target 1건당 2홉 그래프 1개라 검증 전체 3천만 건은 조건마다 못 돌림. 가중 지표는 전체 평가의 불편추정.)
run.json 있으면 건너뜀. 이어 돌리려면 폴더를 지운다.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
import sampling_lib as L


def val_sample(con, window, rate, seed) -> pd.DataFrame:
    """고정 검증 표본. 없으면 만든다."""
    a, b = window; path = L.SAMPLING / "val" / f"{a}_{b}_neg{rate:g}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    lab = L.load_labels(con, window); rng = np.random.default_rng(seed); parts = [lab[lab.y == 1].assign(w=1.0)]
    for day, g in lab[lab.y == 0].groupby("day"):
        k = max(1, int(round(len(g) * rate))); parts.append(g.iloc[rng.choice(len(g), k, replace=False)].assign(w=len(g) / k))
    v = pd.concat(parts).sort_values("edge_id").reset_index(drop=True); path.parent.mkdir(parents=True, exist_ok=True); v.to_parquet(path, index=False)
    L.log(f"검증 표본 생성: 양성 {int(v.y.sum()):,} + 정상 {int((v.y == 0).sum()):,} → {path}")
    return v


def predict(model, store, edge_ids, batch, device):
    import torch
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(edge_ids), batch):
            out.append(model.scores(*store.collate(edge_ids[i:i + batch], device)).cpu().numpy())
    return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True, help="select/<이름> (예: random_r30, none)")
    ap.add_argument("--apply", choices=["loss", "struct"], default="loss")
    ap.add_argument("--reweight", choices=["ipw", "none"], default="none")
    ap.add_argument("--pi-eff", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", default="2022-09-28:2022-10-14"); ap.add_argument("--val-neg-rate", type=float, default=0.02)
    ap.add_argument("--features", default="features")
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32, help="target 수. 2홉 그래프가 target 당 최대 4만 간선이라 크게 못 잡음"); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-train-targets", type=int, default=None, help="에폭당 target 상한(시간 측정·스모크용)")
    ap.add_argument("--device", default="auto"); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--memory", default="12GB")
    a = ap.parse_args(argv)
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    run_name = f"{a.select}_{a.apply}_{a.reweight}_pi{a.pi_eff:g}_s{a.seed}" + ("" if a.features == "features" else f"_{a.features}")
    out = L.SAMPLING / "runs" / run_name
    if (out / "run.json").exists():
        L.log(f"완료된 조건: {run_name}"); return
    out.mkdir(parents=True, exist_ok=True); L.json_write(out / "config.json", vars(a) | {"run_name": run_name, "device": device})
    con = L.connect(a.threads, a.memory); t0 = time.time()

    # --- 학습 표본 ---------------------------------------------------------------
    sel_dir = L.SAMPLING / "select" / a.select
    sel = pd.read_parquet(sel_dir / "selected.parquet"); meta = json.loads((sel_dir / "meta.json").read_text(encoding="utf-8"))
    w = np.where(sel.y == 1, 1.0, sel.ipw.to_numpy() if a.reweight == "ipw" else 1.0)
    n_pos = int((sel.y == 1).sum()); neg_mass = float(w[sel.y == 0].sum())
    pos_weight = a.pi_eff / (1 - a.pi_eff) * neg_mass / n_pos
    L.log(f"{run_name}: target {len(sel):,} (양성 {n_pos:,}, 정상 {len(sel) - n_pos:,}), 정상 가중치 합 {neg_mass:,.0f}, pos_weight={pos_weight:.2f}")

    # --- 그래프 저장소 -----------------------------------------------------------
    exclude = None
    if a.apply == "struct":
        # 학습 구간 정상 중 선택 안 된 것 = 그래프에서 제거할 간선
        lab = L.load_labels(con, L.parse_window(meta["train"]), patterns=False)
        removed = lab[(lab.y == 0) & ~lab.edge_id.isin(sel.edge_id)]
        exclude = out / "excluded_edges.parquet"; removed[["edge_id"]].to_parquet(exclude, index=False)
        L.log(f"구조 샘플링: adjacency 에서 {len(removed):,}건 제거")
    store = L.GraphStore(a.features, exclude, a.threads, a.memory); store.prepare(); store.load()

    # --- 검증 표본 ---------------------------------------------------------------
    val = val_sample(con, L.parse_window(a.val), a.val_neg_rate, 0)
    assert not set(val.edge_id).intersection(sel.edge_id), "학습·검증 겹침"
    con.close()                                   # 학습 중엔 memmap 페이지 캐시가 커지므로 DuckDB 메모리를 비워 둔다

    # --- 학습 ----------------------------------------------------------------------
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = L.make_model(pos_weight).to(device); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    pw = torch.tensor(pos_weight, device=device); ids = sel.edge_id.to_numpy(); ys = sel.y.to_numpy(np.float32); ws = w.astype(np.float32)
    best, stale, history = -np.inf, 0, []
    for epoch in range(a.epochs):
        model.train(); order = np.random.default_rng(a.seed + epoch).permutation(len(ids))
        if a.max_train_targets: order = order[:a.max_train_targets]
        te = time.time(); losses = []
        for i in range(0, len(order), a.batch):
            j = order[i:i + a.batch]
            logits = model(*store.collate(ids[j], device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.as_tensor(ys[j], device=device), pos_weight=pw, reduction="none")
            loss = (loss * torch.as_tensor(ws[j], device=device)).sum() / ws[j].sum()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(loss.item())
            if (i // a.batch) % 200 == 0: L.log(f"  epoch {epoch+1} {i:,}/{len(order):,} loss {np.mean(losses[-200:]):.4f}")
        score = predict(model, store, val.edge_id.to_numpy(), a.batch * 4, device)
        m = L.evaluate(val.y.to_numpy(), score, val.w.to_numpy(), (val.y.to_numpy() == 1) & ~val.in_pattern.to_numpy())
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), train_sec=round(time.time() - te), **m)); L.log(f"epoch {epoch+1}: {m}")
        if m["AP"] > best:
            best, stale = m["AP"], 0; torch.save(model.state_dict(), out / "model.pt")
            pd.DataFrame(dict(edge_id=val.edge_id, y=val.y, w=val.w, score=score)).to_parquet(out / "val_scores.parquet", index=False)
        else:
            stale += 1
            if stale >= a.patience: break
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # --- 결과 ----------------------------------------------------------------------
    bh = max(history, key=lambda h: h["AP"])
    L.json_write(out / "run.json", dict(
        run_name=run_name, select=a.select, method=meta["method"], ratio=meta["ratio"], apply=a.apply, reweight=a.reweight,
        pi_eff=a.pi_eff, pos_weight=pos_weight, seed=a.seed, features=a.features,
        train_rows=len(sel), train_pos=n_pos, train_neg=len(sel) - n_pos, neg_population=meta["neg_population"],
        sampling=dict(daily_keep_min=meta.get("daily_keep_min"), daily_keep_max=meta.get("daily_keep_max")),
        best_epoch=bh["epoch"], val={k: bh[k] for k in ("AP", "F1", "precision", "recall", "threshold", "recall_at_1pct", "unmatched_recall_at_1pct", "ECE")},
        val_rows=len(val), val_neg_rate=a.val_neg_rate, epochs_run=len(history), train_sec_per_epoch=float(np.mean([h["train_sec"] for h in history])),
        gpu_peak_gb=(torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else None), total_sec=round(time.time() - t0)))
    L.log(f"끝: {run_name} best AP {best:.4f} ({(time.time() - t0) / 60:.1f}분)")


if __name__ == "__main__":
    main()
