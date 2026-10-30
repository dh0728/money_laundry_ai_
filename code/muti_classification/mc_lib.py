"""muti_classification 공통 모듈 — 유형(다중클래스) 분류 실험.

01_prepare_laundering.ipynb 가 만든 parquets/<DATASET>/multiclass/ 를 읽고,
분할·잘림·시간 추이 분석 함수를 둔다 (02_split_pattern_analysis.ipynb 에서 부름).

- 큰 원본(transactions.parquet, 1.8억 행)은 01 에서 DuckDB 로 한 번만 스캔한다.
- 여기 함수는 세탁 거래(HI-Large 약 22.5만 건)와 패턴 행(약 14만 건)만 pandas 로 다룬다.

용어
- 시도(attempt) : Patterns.txt 의 BEGIN ~ END 한 묶음. attempt_id 는 01_prepare_parquet 이 붙인 번호
- 구간(period)  : before / train / val / test / gap / after. 분할 창 밖은 before(학습 시작 전), after(테스트 끝 뒤)
- 잘림          : 한 시도의 거래가 둘 이상의 구간에 걸치는 것 (경계 잘림), 또는 데이터 끝(DATA_END) 뒤로 이어지는 것 (끝 잘림)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================ 경로
def find_root() -> Path:
    """datasets/ 또는 parquets/ 가 있는 저장소 루트 (sampling_lib.find_root 와 같은 규칙)"""
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "datasets").is_dir() or (p / "parquets").is_dir():
            return p
    raise FileNotFoundError("datasets/ 또는 parquets/ 가 있는 저장소 루트를 못 찾음")


def base_dir(dataset: str) -> Path:
    """01_prepare_parquet 출력 폴더 (transactions.parquet, patterns.parquet …)"""
    return find_root() / "parquets" / dataset


def mc_dir(dataset: str) -> Path:
    """유형분류 데이터 폴더: parquets/<DATASET>/multiclass/"""
    d = base_dir(dataset) / "multiclass"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(dataset: str, name: str) -> Path:
    """분석 결과(csv, png): code/muti_classification/results/<DATASET>/<name>/"""
    d = Path(__file__).resolve().parent / "results" / dataset / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def q(v) -> str:
    """SQL 문자열 리터럴"""
    return "'" + str(v).replace("'", "''") + "'"


# ============================================================================ 클래스
# 패턴 순서는 bc_lib.PATTERN_ORDER 와 같다.
# TODO: 서비스 TYPE 출력(p_0..p_8) 코드표와 순서를 맞출 것 — 지금 번호는 실험용
PATTERNS = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "RANDOM", "BIPARTITE", "STACK"]
UNMATCHED = "UNMATCHED"                     # 무패턴 세탁 (Is Laundering=1 인데 Patterns.txt 에 없음)
CLASSES = PATTERNS + [UNMATCHED]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# ============================================================================ 분할
# 이진 실험과 같은 분할 (sampling_lib.TRAIN/VAL/TEST, bc_lib.WINDOWS). 양끝 날짜 포함
WINDOWS = {
    "HI-Large": dict(train=("2022-08-08", "2022-10-02"), val=("2022-10-03", "2022-10-16"), test=("2022-10-17", "2022-10-30")),
    "HI-Small": dict(train=("2022-09-01", "2022-09-06"), val=("2022-09-07", "2022-09-08"), test=("2022-09-09", "2022-09-10")),
}
SPLITS = ["train", "val", "test"]
PERIODS = ["before", "train", "val", "test", "gap", "after"]


def _pair(w):
    """('2022-08-08', '2022-10-02') 또는 '2022-08-08:2022-10-02' → 두 문자열"""
    if isinstance(w, str):
        a, b = w.split(":")
        return a, b
    return w


def make_windows(train, val, test) -> dict:
    """각 구간 (시작, 끝) 날짜 → {'train': (Timestamp 시작 0시, Timestamp 끝 다음날 0시)} — 끝 날짜 포함"""
    w = {}
    for k, v in dict(train=train, val=val, test=test).items():
        a, b = _pair(v)
        a, b = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1)
        assert a < b, (k, a, b)
        w[k] = (a, b)
    assert w["train"][1] <= w["val"][0] and w["val"][1] <= w["test"][0], "구간이 겹치거나 순서가 틀림"
    return w


def end_of(day) -> pd.Timestamp:
    """'2022-10-31' → 2022-11-01 00:00 (이 시각 미만이 그 날짜까지)"""
    return pd.Timestamp(day) + pd.Timedelta(days=1)


def assign_period(ts, windows: dict) -> pd.Categorical:
    """시각 → 구간 이름. 창 사이 빈 날은 gap, 학습 시작 전은 before, 테스트 끝 뒤는 after"""
    t = pd.to_datetime(pd.Series(ts)).to_numpy()
    first = min(a for a, _ in windows.values()).to_datetime64()
    last = max(b for _, b in windows.values()).to_datetime64()
    out = np.full(len(t), "gap", dtype=object)
    out[t < first] = "before"
    out[t >= last] = "after"
    for k, (a, b) in windows.items():
        out[(t >= a.to_datetime64()) & (t < b.to_datetime64())] = k
    return pd.Categorical(out, categories=PERIODS)


# ============================================================================ 로드
def load_laundering(dataset: str) -> pd.DataFrame:
    """세탁 거래 전체 (01 출력). pattern_type ∈ CLASSES, attempt_id 는 패턴 거래만"""
    df = pd.read_parquet(mc_dir(dataset) / "laundering.parquet").reset_index(drop=True)
    df["pattern_type"] = pd.Categorical(df["pattern_type"], categories=CLASSES)
    return df


def load_attempts(dataset: str) -> pd.DataFrame:
    """시도 표 (01 출력). 시도 1행: pattern_type, n_rows, n_edges, start_ts, end_ts, duration_h …"""
    df = pd.read_parquet(mc_dir(dataset) / "attempts.parquet").reset_index(drop=True)
    df["pattern_type"] = pd.Categorical(df["pattern_type"], categories=PATTERNS)
    return df


def load_members(dataset: str) -> pd.DataFrame:
    """패턴 파일의 행 그대로 (시도 × 거래). 한 거래가 여러 시도에 들어 있으면 여러 행.
    ts 는 패턴 파일의 시각이라 거래와 매칭 안 된 행(unmatched)도 시각이 있다"""
    df = pd.read_parquet(base_dir(dataset) / "patterns.parquet",
                         columns=["attempt_id", "pattern_type", "edge_id", "ts", "unmatched"]).reset_index(drop=True)
    df["pattern_type"] = pd.Categorical(df["pattern_type"], categories=PATTERNS)
    return df


def load_daily(dataset: str) -> pd.DataFrame:
    """일별 전체 거래 수 / 세탁 거래 수 (01 출력). 정상 = n_all - n_laundering"""
    df = pd.read_parquet(mc_dir(dataset) / "daily_all.parquet")
    df["day"] = pd.to_datetime(df["day"])
    df["n_normal"] = df["n_all"] - df["n_laundering"]
    return df.set_index("day")


# ============================================================================ 1. 구간별 클래스 수
def _with_total(tab: pd.DataFrame) -> pd.DataFrame:
    tab = tab.copy()
    tab["total"] = tab.sum(axis=1)
    tab.loc["ALL"] = tab.sum(axis=0)
    return tab


def split_counts(lau: pd.DataFrame, members: pd.DataFrame, att: pd.DataFrame, windows: dict) -> dict:
    """
    edges          : 클래스 × 구간 엣지 수 (세탁 거래 1건 = 1행, 라벨 기준)
    edges_share    : train/val/test 열마다 클래스 비율(%)
    attempts_start : 패턴 × 구간 시도 수 — 시도의 첫 거래가 속한 구간
    attempts_touch : 패턴 × 구간 시도 수 — 그 구간에 거래가 1건이라도 있는 시도 (한 시도가 여러 열에 셈)
    """
    period = assign_period(lau["ts"], windows)
    edges = (pd.crosstab(lau["pattern_type"], period, dropna=False).reindex(index=CLASSES, columns=PERIODS, fill_value=0)
               .rename_axis(index="pattern_type", columns="period"))
    share = (edges[SPLITS] / edges[SPLITS].sum(axis=0) * 100).round(2)
    share.loc["ALL"] = 100.0

    start = pd.crosstab(att["pattern_type"], assign_period(att["start_ts"], windows), dropna=False)
    start = start.reindex(index=PATTERNS, columns=PERIODS, fill_value=0).rename_axis(index="pattern_type", columns="period")

    m = members[["attempt_id", "pattern_type", "ts"]].assign(period=assign_period(members["ts"], windows))
    touch = (m.drop_duplicates(["attempt_id", "period"])
              .pipe(lambda d: pd.crosstab(d["pattern_type"], d["period"], dropna=False))
              .reindex(index=PATTERNS, columns=PERIODS, fill_value=0))
    return dict(edges=_with_total(edges), edges_share=share, attempts_start=_with_total(start), attempts_touch=touch)


# ============================================================================ 2. 구간 경계에서 잘리는 시도
BOUNDARIES = [("before", "train"), ("train", "val"), ("val", "test"), ("test", "after")]


def attempt_periods(members: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """시도 × 구간 행 수 (패턴 파일 행 기준). 열 순서 = PERIODS"""
    m = members[["attempt_id", "ts"]].assign(period=assign_period(members["ts"], windows))
    cnt = pd.crosstab(m["attempt_id"], m["period"], dropna=False)
    return cnt.reindex(columns=PERIODS, fill_value=0)


def boundary_cuts(members: pd.DataFrame, windows: dict) -> dict:
    """
    by_pattern    : 패턴별 시도 수, 둘 이상 구간에 걸친 시도 수(cut)·비율, 경계별 걸친 시도 수,
                    걸친 시도에서 '첫 구간 밖' 행 비율의 중앙값 (얼마나 크게 잘리나)
    edge_exposure : train/val/test 의 패턴 엣지 중 같은 시도의 거래가 앞 구간(with_earlier) / 뒤 구간(with_later)에도 있는 비율(%)
                    - test 의 with_earlier : 같은 시도의 일부를 학습·검증에서 봄 → 누수 후보
                    - train 의 with_later  : 학습 그래프(학습 끝까지)에서는 시도 뒷부분이 안 보임 → 잘린 맥락으로 학습
    per_attempt   : 시도별 상세 (attempt_id, pattern_type, n_periods, cut, 경계 플래그, outside_share)
    """
    cnt = attempt_periods(members, windows)
    has = cnt.to_numpy() > 0
    ptype = members.groupby("attempt_id", observed=True)["pattern_type"].first().reindex(cnt.index)

    df = pd.DataFrame({"pattern_type": ptype.astype(str).to_numpy()}, index=cnt.index)
    df["n_periods"] = has.sum(axis=1)
    df["cut"] = df["n_periods"] > 1
    col = {p: i for i, p in enumerate(PERIODS)}
    for a, b in BOUNDARIES:
        df[f"{a}|{b}"] = has[:, col[a]] & has[:, col[b]]
    first = has.argmax(axis=1)                                   # 첫 구간 (PERIODS 순서)
    last = has.shape[1] - 1 - has[:, ::-1].argmax(axis=1)        # 마지막 구간
    rows = cnt.to_numpy()
    df["outside_share"] = 1 - rows[np.arange(len(rows)), first] / rows.sum(axis=1)
    df["first_period"] = np.array(PERIODS)[first]
    df["last_period"] = np.array(PERIODS)[last]

    bnames = [f"{a}|{b}" for a, b in BOUNDARIES]
    g = df.groupby("pattern_type")
    tab = g.agg(attempts=("cut", "size"), cut=("cut", "sum"), **{b: (b, "sum") for b in bnames})
    tab["cut_%"] = (tab["cut"] / tab["attempts"] * 100).round(2)
    tab["outside_share_med(cut)"] = df[df["cut"]].groupby("pattern_type")["outside_share"].median().round(3)
    tab = tab.reindex([p for p in PATTERNS if p in tab.index])
    tab.loc["ALL"] = dict(attempts=len(df), cut=int(df["cut"].sum()), **{b: int(df[b].sum()) for b in bnames},
                          **{"cut_%": round(df["cut"].mean() * 100, 2),
                             "outside_share_med(cut)": round(df.loc[df["cut"], "outside_share"].median(), 3)})
    tab = tab[["attempts", "cut", "cut_%", *bnames, "outside_share_med(cut)"]]

    # 엣지 관점
    m = members[["attempt_id", "pattern_type", "ts"]].assign(period=assign_period(members["ts"], windows))
    m["pi"] = m["period"].cat.codes
    m["a_first"] = m["attempt_id"].map(pd.Series(first, index=cnt.index))
    m["a_last"] = m["attempt_id"].map(pd.Series(last, index=cnt.index))
    m["with_earlier"] = m["a_first"] < m["pi"]
    m["with_later"] = m["a_last"] > m["pi"]
    ex = (m[m["period"].isin(SPLITS)]
          .groupby(["period", "pattern_type"], observed=True)
          .agg(n=("ts", "size"), with_earlier=("with_earlier", "mean"), with_later=("with_later", "mean")))
    ex[["with_earlier", "with_later"]] = (ex[["with_earlier", "with_later"]] * 100).round(2)
    ex = ex.rename(columns={"with_earlier": "with_earlier_%", "with_later": "with_later_%"})
    return dict(by_pattern=tab, edge_exposure=ex, per_attempt=df)


# ============================================================================ 3. 데이터 끝(DATA_END)에서 잘리는 시도
def truncation(members: pd.DataFrame, att: pd.DataFrame, lau: pd.DataFrame, data_end) -> dict:
    """DATA_END(그 날짜 포함)까지만 쓸 때.
    attempts   : 패턴별 complete(끝까지 안에 있음) / truncated(안에서 시작해 밖으로 이어짐) / started_after(밖에서 시작) 시도 수,
                 truncated 비율, 잘려 나가는 행 수, truncated 시도에서 잃는 행 비율(중앙값·최대)
    edges      : 클래스별 세탁 엣지 수 — DATA_END 까지 / 이후
    last_seen  : 클래스별 마지막 거래 시각
    detail     : 시도별 상세
    """
    cut = end_of(data_end)
    a = att[["attempt_id", "pattern_type", "n_rows", "start_ts", "end_ts", "duration_h"]].copy()
    a["status"] = np.select([a["end_ts"] < cut, a["start_ts"] >= cut], ["complete", "started_after"], "truncated")
    after = members.loc[members["ts"] >= cut].groupby("attempt_id").size()
    a["rows_after"] = a["attempt_id"].map(after).fillna(0).astype(int)
    a["lost_share"] = a["rows_after"] / a["n_rows"]

    tab = pd.crosstab(a["pattern_type"], a["status"], dropna=False).reindex(
        index=PATTERNS, columns=["complete", "truncated", "started_after"], fill_value=0).rename_axis(columns=None)
    tab.insert(0, "attempts", tab.sum(axis=1))
    tab["truncated_%"] = (tab["truncated"] / tab["attempts"] * 100).round(2)
    tab["rows_after"] = a.groupby("pattern_type", observed=False)["rows_after"].sum().reindex(PATTERNS)
    tr = a[a["status"] == "truncated"].groupby("pattern_type", observed=False)["lost_share"]
    tab["lost_share_med(trunc)"] = tr.median().reindex(PATTERNS).round(3)
    tab["lost_share_max(trunc)"] = tr.max().reindex(PATTERNS).round(3)
    tab.loc["ALL"] = dict(attempts=len(a), **{s: int((a["status"] == s).sum()) for s in ["complete", "truncated", "started_after"]},
                          **{"truncated_%": round((a["status"] == "truncated").mean() * 100, 2),
                             "rows_after": int(a["rows_after"].sum()),
                             "lost_share_med(trunc)": round(a.loc[a["status"] == "truncated", "lost_share"].median(), 3),
                             "lost_share_max(trunc)": round(a.loc[a["status"] == "truncated", "lost_share"].max(), 3)})

    inside = lau["ts"] < cut
    edges = pd.DataFrame({"until_end": lau.loc[inside, "pattern_type"].value_counts(),
                          "after_end": lau.loc[~inside, "pattern_type"].value_counts()}).reindex(CLASSES).fillna(0).astype(int)
    edges["after_%"] = (edges["after_end"] / (edges["until_end"] + edges["after_end"]) * 100).round(2)
    edges.loc["ALL"] = [int(inside.sum()), int((~inside).sum()), round((~inside).mean() * 100, 2)]
    edges[["until_end", "after_end"]] = edges[["until_end", "after_end"]].astype(int)
    last_seen = lau.groupby("pattern_type", observed=False)["ts"].agg(["min", "max"]).reindex(CLASSES)
    return dict(attempts=tab, edges=edges, last_seen=last_seen, detail=a)


def duration_table(att: pd.DataFrame) -> pd.DataFrame:
    """패턴별 시도 길이(일)·크기(행 수) 분포"""
    d = att.assign(duration_d=att["duration_h"] / 24)
    g = d.groupby("pattern_type", observed=False)
    out = pd.concat({
        "duration_d": g["duration_d"].quantile([0.5, 0.9, 0.99, 1.0]).unstack(),
        "n_rows": g["n_rows"].quantile([0.5, 0.9, 1.0]).unstack(),
    }, axis=1).round(2)
    out.columns = [f"{a}_p{int(b*100)}" if b < 1 else f"{a}_max" for a, b in out.columns]
    return out.reindex(PATTERNS)


# ============================================================================ 4. 여러 분할 비교
def compare_splits(configs: dict, lau, members, att, data_end) -> pd.DataFrame:
    """configs = {'이름': (train, val, test)} → 분할마다 한 행 요약.
    n_<split>          : 세탁 엣지 수
    min_<split>        : 그 구간에서 가장 적은 클래스의 엣지 수 (클래스)
    unmatched_<split>% : 그 구간의 무패턴 비율
    cut_%              : 둘 이상 구간에 걸친 시도 비율
    test_with_earlier% : 테스트 패턴 엣지 중 같은 시도가 앞 구간에도 있는 비율 (누수 후보)
    truncated_%        : DATA_END 에서 끝이 잘리는 시도 비율"""
    rows = []
    tr = truncation(members, att, lau, data_end)["attempts"].loc["ALL"]
    for name, (tr_w, va_w, te_w) in configs.items():
        w = make_windows(tr_w, va_w, te_w)
        sc = split_counts(lau, members, att, w)["edges"]
        bc = boundary_cuts(members, w)
        r = dict(config=name, train=f"{_pair(tr_w)[0]}~{_pair(tr_w)[1]}", val=f"{_pair(va_w)[0]}~{_pair(va_w)[1]}",
                 test=f"{_pair(te_w)[0]}~{_pair(te_w)[1]}")
        for s in SPLITS:
            col = sc.loc[CLASSES, s]
            r[f"n_{s}"] = int(col.sum())
            r[f"min_{s}"] = f"{int(col.min())} ({col.idxmin()})"
            r[f"unmatched_{s}%"] = round(col[UNMATCHED] / max(col.sum(), 1) * 100, 1)
        r["cut_%"] = bc["by_pattern"].loc["ALL", "cut_%"]
        ex = bc["edge_exposure"]
        if "test" in ex.index.get_level_values(0):
            t = ex.loc["test"]
            r["test_with_earlier%"] = round((t["n"] * t["with_earlier_%"]).sum() / t["n"].sum(), 2)
        r["truncated_%"] = tr["truncated_%"]
        rows.append(r)
    return pd.DataFrame(rows).set_index("config")


# ============================================================================ 5. 그림 (영문 라벨 — GPU 서버에 한글 폰트가 없을 수 있음)
def _mark(ax, windows, data_end=None):
    for k, (a, b) in windows.items():
        ax.axvspan(a, b, color={"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}[k], alpha=0.08, lw=0)
    if data_end is not None:
        ax.axvline(end_of(data_end), color="#D62728", lw=1, ls="--")


def plot_daily_classes(lau, windows, data_end=None, daily=None, path=None):
    """일별 세탁 엣지 수 — 첫 칸: 정상·세탁 전체(log), 나머지 9칸: 클래스별 (y축 각자). 음영 = train/val/test, 빨간 점선 = DATA_END"""
    import matplotlib.pyplot as plt
    day = lau["ts"].dt.floor("D")
    d = (lau.assign(day=day).pivot_table(index="day", columns="pattern_type", values="edge_id", aggfunc="size", observed=False)
            .reindex(columns=CLASSES).fillna(0))
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"), fill_value=0)
    fig, axes = plt.subplots(5, 2, figsize=(15, 17), sharex=True)
    axes = axes.ravel()
    ax = axes[0]
    if daily is not None:
        ax.plot(daily.index, daily["n_normal"], color="#7F7F7F", lw=1, label="normal")
    ax.plot(d.index, d.sum(axis=1), color="black", lw=1, label="laundering (all)")
    ax.set_yscale("log"); ax.set_title("all transactions per day (log)"); ax.legend(fontsize=8)
    _mark(ax, windows, data_end)
    for ax, c in zip(axes[1:], CLASSES):
        ax.plot(d.index, d[c], lw=1, color="#D62728" if c == UNMATCHED else "#1F77B4")
        ax.set_title(f"{c}  (n={int(d[c].sum()):,})"); _mark(ax, windows, data_end)
    for ax in axes[-2:]:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig


def plot_attempt_starts(att, windows, data_end=None, path=None):
    """패턴별 일별 시도 시작 수와, 그날 시작한 시도의 길이(일) 중앙값"""
    import matplotlib.pyplot as plt
    a = att.assign(day=att["start_ts"].dt.floor("D"), duration_d=att["duration_h"] / 24)
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), sharex=True)
    rng = pd.date_range(a["day"].min(), a["day"].max(), freq="D")
    for ax, p in zip(axes.ravel(), PATTERNS):
        s = a[a["pattern_type"] == p].groupby("day")
        n = s.size().reindex(rng, fill_value=0)
        ax.bar(n.index, n.to_numpy(), width=1, color="#1F77B4")
        ax2 = ax.twinx()
        ax2.plot(s["duration_d"].median().reindex(rng).index, s["duration_d"].median().reindex(rng).to_numpy(),
                 ".", ms=3, color="#FF7F0E")
        ax2.set_ylabel("median duration (d)", fontsize=8, color="#FF7F0E")
        ax.set_title(f"{p}: attempt starts per day (bars), duration (dots)")
        _mark(ax, windows, data_end)
    for ax in axes[-1]:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    return fig
