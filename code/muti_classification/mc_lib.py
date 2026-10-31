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


# ============================================================================ 3b. 라벨 겹침(누수) · 성숙도
def label_overlap(members: pd.DataFrame, att: pd.DataFrame, windows: dict) -> dict:
    """같은 시도의 엣지가 다른 분할의 '라벨'로 쓰였는지. edge_exposure 의 with_earlier 는 검증 구간 맥락까지 섞여 있어서 따로 본다.
    edges    : 분할 × 패턴. 그 분할 패턴 엣지 중
               new_%      = 시도가 같은 분할에서 시작 (앞 구간에 흔적 없음 → 깨끗한 평가 묶음)
               in_train_% = 같은 시도가 학습 구간에도 엣지가 있음 (val·test 에서 = 라벨 누수 후보: 학습 때 이 시도의 다른 엣지 정답을 봄)
               in_val_%   = 같은 시도가 검증 구간에도 엣지가 있음 (test 에서 = 모델 선택에 쓰인 시도)
               before_%   = 시도가 학습 시작 전(before)에 시작 — 그 부분은 라벨 없이 맥락으로만 보임
    attempts : 패턴 × (지표, 분할) 시도 수. touch = 그 분할에 엣지가 있는 시도, also_train = 그중 학습에도 있는 시도, new = 그 분할에서 시작"""
    cnt = attempt_periods(members, windows)
    has = pd.DataFrame(cnt.to_numpy() > 0, index=cnt.index, columns=PERIODS)
    start_p = assign_period(att.set_index("attempt_id").reindex(cnt.index)["start_ts"], windows)
    a = has.assign(start_period=np.asarray(start_p))

    m = members[["attempt_id", "pattern_type", "ts"]].assign(period=assign_period(members["ts"], windows))
    m = m[m["period"].isin(SPLITS)].join(a, on="attempt_id")
    m["new"] = m["start_period"] == m["period"].astype(str)
    m["before_start"] = m["start_period"] == "before"
    g = m.groupby(["period", "pattern_type"], observed=True)
    edges = g.agg(n=("ts", "size"), **{"new_%": ("new", "mean"), "in_train_%": ("train", "mean"),
                                       "in_val_%": ("val", "mean"), "before_%": ("before_start", "mean")})
    pct = [c for c in edges.columns if c.endswith("%")]
    edges[pct] = (edges[pct] * 100).round(2)
    tot = m.groupby("period", observed=True).agg(n=("ts", "size"), **{"new_%": ("new", "mean"), "in_train_%": ("train", "mean"),
                                                                      "in_val_%": ("val", "mean"), "before_%": ("before_start", "mean")})
    tot[pct] = (tot[pct] * 100).round(2)
    tot.index = pd.MultiIndex.from_tuples([(p, "ALL") for p in tot.index], names=edges.index.names)
    edges = pd.concat([edges, tot])
    order = [(sp, pt) for sp in SPLITS for pt in PATTERNS + ["ALL"] if (sp, pt) in edges.index]
    edges = edges.reindex(order)

    rows = []
    ptype = members.groupby("attempt_id", observed=True)["pattern_type"].first().reindex(cnt.index).astype(str)
    for s in SPLITS:
        t = a[a[s]]
        r = pd.DataFrame({"touch": t.groupby(ptype[t.index]).size(),
                          "also_train": t[t["train"]].groupby(ptype[t[t["train"]].index]).size(),
                          "new": t[t["start_period"] == s].groupby(ptype[t[t["start_period"] == s].index]).size()})
        rows.append(r.reindex(PATTERNS).fillna(0).astype(int).assign(split=s))
    attempts = (pd.concat(rows).reset_index(names="pattern_type")
                  .pivot(index="pattern_type", columns="split", values=["touch", "also_train", "new"])
                  .reindex(index=PATTERNS))
    attempts = attempts.reindex(columns=pd.MultiIndex.from_product([["touch", "also_train", "new"], SPLITS]))
    return dict(edges=edges, attempts=attempts)


AGE_BINS = [0, 1, 3, 7, 14, 21, 28, 35, 49, np.inf]


def edge_age(lau: pd.DataFrame, att: pd.DataFrame, windows: dict) -> dict:
    """성숙도 = 엣지 시각 − 시도 시작 시각 (일). 진행 중인 패턴을 얼마나 일찍 맞히는지 볼 때의 층화 축.
    quantiles : 분할 × 패턴 나이 분위 (p10/p50/p90)
    bins      : 분할별 나이 구간 × 패턴 엣지 수 (AGE_BINS)"""
    p = lau[lau["attempt_id"].notna()].copy()
    p["attempt_id"] = p["attempt_id"].astype(np.int64)
    p = p.join(att.set_index("attempt_id")[["start_ts"]], on="attempt_id")
    p["age_d"] = (p["ts"] - p["start_ts"]).dt.total_seconds() / 86400
    p["period"] = assign_period(p["ts"], windows)
    p = p[p["period"].isin(SPLITS)]
    q = (p.groupby(["period", "pattern_type"], observed=True)["age_d"].quantile([0.1, 0.5, 0.9]).unstack().round(1))
    q.columns = ["age_p10", "age_p50", "age_p90"]
    p["age_bin"] = pd.cut(p["age_d"], AGE_BINS, right=False)
    bins = p.pivot_table(index=["period", "age_bin"], columns="pattern_type", values="edge_id", aggfunc="size", observed=False).fillna(0).astype(int)
    bins = bins[[c for c in PATTERNS if c in bins.columns]]
    bins = bins[bins.index.get_level_values(0).isin(SPLITS)]
    return dict(quantiles=q, bins=bins)


# ============================================================================ 3c. 시도 안의 순서 · 진행 중 모습
LOOKALIKE = {"GATHER-SCATTER": "FAN-IN", "SCATTER-GATHER": "FAN-OUT", "STACK": "BIPARTITE",
             "CYCLE": "path", "RANDOM": "path"}


def node_roles(lau: pd.DataFrame) -> pd.DataFrame:
    """시도 × 계좌. 시도 안에서 받은/보낸 엣지 수와 첫·마지막 시각.
    중간 계좌(intermediate) = 시도 안에서 받기도 하고 보내기도 한 계좌. hub = 시도 안 in+out 이 가장 큰 계좌(동률이면 아무거나)"""
    p = lau[lau["attempt_id"].notna()][["attempt_id", "pattern_type", "src", "dst", "ts"]].copy()
    p["attempt_id"] = p["attempt_id"].astype(np.int64)
    out = (p.groupby(["attempt_id", "src"]).agg(n_out=("ts", "size"), first_out=("ts", "min"), last_out=("ts", "max"))
             .rename_axis(["attempt_id", "node"]))
    inn = (p.groupby(["attempt_id", "dst"]).agg(n_in=("ts", "size"), first_in=("ts", "min"), last_in=("ts", "max"))
             .rename_axis(["attempt_id", "node"]))
    r = out.join(inn, how="outer").reset_index()
    r[["n_out", "n_in"]] = r[["n_out", "n_in"]].fillna(0).astype(int)
    r["intermediate"] = (r["n_in"] > 0) & (r["n_out"] > 0)
    r["deg"] = r["n_in"] + r["n_out"]
    r["is_hub"] = r.index.isin(r.groupby("attempt_id")["deg"].idxmax())
    r = r.join(p.groupby("attempt_id")["pattern_type"].first(), on="attempt_id")
    return r


def phase_order(lau: pd.DataFrame) -> dict:
    """시도 안에서 '받기 → 보내기' 순서가 지켜지는지 (GATHER-SCATTER 가 모으기 단계 → 뿌리기 단계로 생성됐는지 확인).
    중간 계좌마다 ordered = 마지막 입금 ≤ 첫 출금 (모두 받은 뒤에 보냄), overlap = 첫 출금 < 마지막 입금.
    by_pattern : 패턴별
        attempts, has_inter_%        : 중간 계좌가 있는 시도 비율 (FAN-IN/OUT·BIPARTITE 는 원래 0 에 가까워야 함)
        inter_nodes, node_ordered_%  : 중간 계좌 중 ordered 비율
        attempt_all_ordered_%        : 중간 계좌가 모두 ordered 인 시도 비율 (중간 계좌 있는 시도 중)
        hub_* (hub 가 중간 계좌인 시도) : hub 의 받기 구간 길이 · 보내기 구간 길이 · 사이 간격(첫 출금 − 마지막 입금, 음수 = 겹침) 중앙값(일),
                                     hub_in_share_med = hub 의 첫 출금 전까지 도착한 시도 엣지 비율 (G-S 면 모으기 단계 비중)
    hub        : 시도별 hub 상세"""
    r = node_roles(lau)
    inter = r[r["intermediate"]].copy()
    inter["ordered"] = inter["last_in"] <= inter["first_out"]
    att_ord = inter.groupby("attempt_id").agg(pattern_type=("pattern_type", "first"), all_ordered=("ordered", "all"))
    n_att = r.groupby("attempt_id")["pattern_type"].first()

    hub = r[r["is_hub"] & r["intermediate"]].copy()
    day = pd.Timedelta(days=1)
    hub["in_span_d"] = (hub["last_in"] - hub["first_in"]) / day
    hub["out_span_d"] = (hub["last_out"] - hub["first_out"]) / day
    hub["gap_d"] = (hub["first_out"] - hub["last_in"]) / day
    # 시도 엣지 중 hub 첫 출금 이전 도착 비율
    p = lau[lau["attempt_id"].notna()][["attempt_id", "ts"]].copy()
    p["attempt_id"] = p["attempt_id"].astype(np.int64)
    p = p.join(hub.set_index("attempt_id")["first_out"], on="attempt_id", how="inner")
    hub = hub.join((p["ts"] < p["first_out"]).groupby(p["attempt_id"]).mean().rename("in_share"), on="attempt_id")

    rows = []
    for pt in PATTERNS:
        a_ids = n_att[n_att == pt].index
        it = inter[inter["pattern_type"] == pt]
        ao = att_ord[att_ord["pattern_type"] == pt]
        h = hub[hub["pattern_type"] == pt]
        rows.append({"pattern_type": pt, "attempts": len(a_ids),
                     "has_inter_%": round(len(ao) / max(len(a_ids), 1) * 100, 2),
                     "inter_nodes": len(it),
                     "node_ordered_%": round(it["ordered"].mean() * 100, 2) if len(it) else np.nan,
                     "attempt_all_ordered_%": round(ao["all_ordered"].mean() * 100, 2) if len(ao) else np.nan,
                     "hub_inter_attempts": len(h),
                     "hub_in_span_d_med": round(h["in_span_d"].median(), 2) if len(h) else np.nan,
                     "hub_out_span_d_med": round(h["out_span_d"].median(), 2) if len(h) else np.nan,
                     "hub_gap_d_med": round(h["gap_d"].median(), 2) if len(h) else np.nan,
                     "hub_gap_neg_%": round((h["gap_d"] < 0).mean() * 100, 2) if len(h) else np.nan,
                     "hub_in_share_med": round(h["in_share"].median(), 3) if len(h) else np.nan})
    return dict(by_pattern=pd.DataFrame(rows).set_index("pattern_type"), hub=hub)


def flat_view(lau: pd.DataFrame, windows: dict) -> dict:
    """진행 중 모습: 어떤 시점까지 보이는 시도 엣지만으로 그래프를 그렸을 때 중간 계좌(받고 보낸 계좌)가 아직 하나도 없으면 'flat'.
    flat 인 GATHER-SCATTER 는 FAN-IN, SCATTER-GATHER 는 FAN-OUT, STACK 은 BIPARTITE 와 구조가 같다 (LOOKALIKE).
    시도마다 t_inter = 중간 계좌가 처음 생기는 시각 = min over 계좌 max(첫 입금, 첫 출금).
      causal_flat   : 엣지 자기 시각 기준 (그 엣지까지 보임) — 실시간에 가장 불리한 경우
      at_end_flat   : 엣지가 속한 분할 끝 기준 (실험 그래프 = 분할 끝까지 누적) — 지금 실험 조건
    by_split : 분할 × 패턴 엣지 중 flat 비율(%)
    never    : 패턴별 끝까지 중간 계좌가 안 생기는 시도 비율 (그 패턴은 원래 flat)"""
    r = node_roles(lau)
    r["t_node"] = r[["first_in", "first_out"]].max(axis=1).where(r["intermediate"])
    t_inter = r.groupby("attempt_id")["t_node"].min()
    p = lau[lau["attempt_id"].notna()][["attempt_id", "pattern_type", "ts"]].copy()
    p["attempt_id"] = p["attempt_id"].astype(np.int64)
    p["t_inter"] = p["attempt_id"].map(t_inter)
    p["period"] = assign_period(p["ts"], windows)
    end = pd.Series({k: b for k, (_, b) in windows.items()})
    p["split_end"] = p["period"].astype(str).map(end)
    p["causal_flat"] = p["t_inter"].isna() | (p["t_inter"] > p["ts"])
    p["at_end_flat"] = p["t_inter"].isna() | (p["t_inter"] >= p["split_end"])
    q = p[p["period"].isin(SPLITS)]
    by = q.groupby(["period", "pattern_type"], observed=True).agg(n=("ts", "size"), causal_flat=("causal_flat", "mean"),
                                                                    at_end_flat=("at_end_flat", "mean"))
    by[["causal_flat", "at_end_flat"]] = (by[["causal_flat", "at_end_flat"]] * 100).round(2)
    by = by.rename(columns={"causal_flat": "causal_flat_%", "at_end_flat": "at_end_flat_%"})
    by["looks_like"] = [LOOKALIKE.get(pt, "-") for pt in by.index.get_level_values(1)]
    never = (t_inter.isna().groupby(r.groupby("attempt_id")["pattern_type"].first()).mean() * 100).round(2)
    return dict(by_split=by, never=never.reindex(PATTERNS).rename("never_inter_%"))


# ============================================================================ 3d. 세탁만 그래프의 구조 (0단계 점검 — 03 노트북)
NB_KINDS = ["isolated", "only_unmatched", "only_pattern", "mixed"]


def _split_targets(lau, windows):
    """(분할 이름, 그래프 = 분할 끝 이전 세탁 엣지 전부, 대상 = 그 분할 엣지 마스크)"""
    for s in SPLITS:
        end = windows[s][1]
        view = lau[lau["ts"] < end]
        yield s, view, view["ts"] >= windows[s][0]


def neighbor_context(lau: pd.DataFrame, windows: dict) -> dict:
    """세탁만 그래프(분할 끝까지 누적)에서 대상 엣지의 양끝 계좌에 붙은 '다른' 엣지가 무엇인지.
    isolated       : 양끝 계좌에 다른 세탁 엣지가 없음 (그 엣지 하나만 덩그러니)
    only_unmatched : 다른 엣지가 전부 UNMATCHED
    only_pattern   : 다른 엣지가 전부 패턴 엣지
    mixed          : 둘 다 있음
    by_class  : 분할 × 클래스 → 위 4종 비율(%) + 다른 엣지 수 중앙값
    given_kind: 분할 × 이웃 종류 → 클래스 분포(%) — 'isolated 면 거의 UNMATCHED' 같은 쉬운 규칙이 있는지 (행 합 100)"""
    by_class, given = [], []
    for s, v, tgt in _split_targets(lau, windows):
        is_pat = (v["pattern_type"] != UNMATCHED).to_numpy().astype(np.int64)
        src, dst = v["src"].to_numpy(), v["dst"].to_numpy()
        n = int(max(src.max(), dst.max())) + 1
        deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
        dpat = np.bincount(src, weights=is_pat, minlength=n) + np.bincount(dst, weights=is_pat, minlength=n)
        loop = src == dst
        # 다른 엣지 수: 양끝 차수 합 − 자기 자신(2). 셀프루프는 한 계좌 차수만
        other = np.where(loop, deg[src] - 2, deg[src] + deg[dst] - 2)
        opat = np.where(loop, dpat[src] - 2 * is_pat, dpat[src] + dpat[dst] - 2 * is_pat)
        ounm = other - opat
        kind = np.select([other == 0, opat == 0, ounm == 0], ["isolated", "only_unmatched", "only_pattern"], "mixed")
        d = pd.DataFrame({"pattern_type": v["pattern_type"].to_numpy(), "kind": kind, "other": other})[tgt.to_numpy()]
        t = pd.crosstab(d["pattern_type"], d["kind"], normalize="index").reindex(index=CLASSES, columns=NB_KINDS).fillna(0) * 100
        t["other_med"] = d.groupby("pattern_type", observed=False)["other"].median().reindex(CLASSES)
        t["n"] = d["pattern_type"].value_counts().reindex(CLASSES).fillna(0).astype(int)
        by_class.append(t.round(2).assign(split=s))
        g = pd.crosstab(d["kind"], d["pattern_type"], normalize="index").reindex(index=NB_KINDS, columns=CLASSES).fillna(0) * 100
        g["n"] = d["kind"].value_counts().reindex(NB_KINDS).fillna(0).astype(int)
        given.append(g.round(2).assign(split=s))
    by_class = pd.concat(by_class).reset_index().set_index(["split", "pattern_type"])
    given = pd.concat(given).rename_axis("kind").reset_index().set_index(["split", "kind"])
    return dict(by_class=by_class, given_kind=given)


def components(lau: pd.DataFrame, cutoff) -> dict:
    """세탁만 그래프(cutoff 날짜까지, 방향 무시)의 연결 요소.
    comp      : 요소별 n_edges, n_nodes, n_attempts, n_unmatched, n_pattern, kind
                kind = unmatched_only / single_attempt(시도 1개 + UNMATCHED 0) / single_attempt+unm / multi_attempt
    summary   : 요소 수, 크기 분위, 가장 큰 요소, 종류별 요소 수·엣지 비율
    by_pattern: 패턴별 시도 중 '자기 요소에 다른 시도가 있음' / 'UNMATCHED 가 섞임' 비율, 요소 엣지 수 중앙값
    by_class  : 클래스별 엣지가 어떤 종류 요소에 있나 (%)"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    v = lau[lau["ts"] < end_of(cutoff)].copy()
    nodes, inv = np.unique(np.concatenate([v["src"].to_numpy(), v["dst"].to_numpy()]), return_inverse=True)
    m = len(v)
    a, b = inv[:m], inv[m:]
    g = coo_matrix((np.ones(m, dtype=np.int8), (a, b)), shape=(len(nodes), len(nodes)))
    n_comp, label = connected_components(g, directed=False)
    v["comp"] = label[a]
    v["is_pat"] = v["pattern_type"] != UNMATCHED
    comp = v.groupby("comp").agg(n_edges=("ts", "size"), n_attempts=("attempt_id", "nunique"),
                                 n_pattern=("is_pat", "sum"))
    comp["n_unmatched"] = comp["n_edges"] - comp["n_pattern"]
    comp["n_nodes"] = pd.Series(np.bincount(label, minlength=n_comp)).reindex(comp.index)
    comp["kind"] = np.select([comp["n_attempts"] == 0, (comp["n_attempts"] == 1) & (comp["n_unmatched"] == 0),
                              comp["n_attempts"] == 1], ["unmatched_only", "single_attempt", "single_attempt+unm"], "multi_attempt")
    kinds = ["unmatched_only", "single_attempt", "single_attempt+unm", "multi_attempt"]
    q = comp["n_edges"].quantile([0.5, 0.9, 0.99]).to_dict()
    top = comp.sort_values("n_edges", ascending=False).iloc[0]
    k = comp.groupby("kind").agg(components=("n_edges", "size"), edges=("n_edges", "sum")).reindex(kinds).fillna(0)
    k["edges_%"] = (k["edges"] / m * 100).round(2)
    summary = dict(cutoff=str(cutoff), edges=m, nodes=len(nodes), components=n_comp,
                   comp_edges_p50=q[0.5], comp_edges_p90=q[0.9], comp_edges_p99=q[0.99],
                   largest_edges=int(top["n_edges"]), largest_nodes=int(top["n_nodes"]),
                   largest_attempts=int(top["n_attempts"]), largest_edges_share_pct=round(top["n_edges"] / m * 100, 2))

    pa = v[v["is_pat"]].groupby("attempt_id").agg(pattern_type=("pattern_type", "first"), comp=("comp", "first"))
    pa = pa.join(comp[["n_attempts", "n_unmatched", "n_edges"]], on="comp")
    bp = pa.groupby("pattern_type", observed=True).agg(attempts=("comp", "size"),
                                                         shares_comp_with_other_attempt=("n_attempts", lambda x: (x > 1).mean() * 100),
                                                         comp_has_unmatched=("n_unmatched", lambda x: (x > 0).mean() * 100),
                                                         comp_edges_med=("n_edges", "median"))
    bp = bp.round(2).reindex(PATTERNS)
    bp = bp.rename(columns={"shares_comp_with_other_attempt": "other_attempt_in_comp_%", "comp_has_unmatched": "unmatched_in_comp_%"})
    v = v.join(comp["kind"], on="comp")
    bc = (pd.crosstab(v["pattern_type"], v["kind"], normalize="index") * 100).reindex(index=CLASSES, columns=kinds).fillna(0).round(2)
    return dict(comp=comp, summary=pd.Series(summary), kinds=k, by_pattern=bp, by_class=bc)


def shared_accounts(lau: pd.DataFrame, cutoff=None) -> dict:
    """한 계좌가 여러 시도(또는 시도 + UNMATCHED)에 쓰이는지.
    accounts : 계좌별 시도 수 분포 (패턴 계좌만)
    by_pattern: 패턴별 시도 중 다른 시도와 계좌를 공유 / 다른 '유형' 시도와 공유 / UNMATCHED 엣지가 있는 계좌를 포함 비율(%)"""
    v = lau if cutoff is None else lau[lau["ts"] < end_of(cutoff)]
    ends = pd.concat([v[["src", "attempt_id", "pattern_type"]].rename(columns={"src": "node"}),
                      v[["dst", "attempt_id", "pattern_type"]].rename(columns={"dst": "node"})])
    unm_nodes = set(ends.loc[ends["pattern_type"] == UNMATCHED, "node"].unique())
    pe = ends[ends["attempt_id"].notna()].drop_duplicates(["node", "attempt_id"]).copy()
    pe["pattern_type"] = pe["pattern_type"].astype(str)
    per_node = pe.groupby("node").agg(n_att=("attempt_id", "nunique"), n_types=("pattern_type", "nunique"))
    acc = per_node["n_att"].clip(upper=5).value_counts().sort_index().rename("accounts")
    acc.index = [str(i) if i < 5 else "5+" for i in acc.index]
    pe = pe.join(per_node, on="node")
    pe["in_unm"] = pe["node"].isin(unm_nodes)
    pa = pe.groupby("attempt_id").agg(pattern_type=("pattern_type", "first"), shared=("n_att", lambda x: (x > 1).any()),
                                      shared_other_type=("n_types", lambda x: (x > 1).any()), has_unm_node=("in_unm", "any"))
    bp = pa.groupby("pattern_type").agg(attempts=("shared", "size"), **{"shared_%": ("shared", "mean"),
                                        "shared_other_type_%": ("shared_other_type", "mean"), "has_unmatched_account_%": ("has_unm_node", "mean")})
    for c in ["shared_%", "shared_other_type_%", "has_unmatched_account_%"]:
        bp[c] = (bp[c] * 100).round(2)
    return dict(accounts=acc, by_pattern=bp.reindex(PATTERNS))


# ============================================================================ 4. 여러 분할 비교
def compare_splits(configs: dict, lau, members, att, data_end) -> pd.DataFrame:
    """configs = {'이름': (train, val, test)} → 분할마다 한 행 요약.
    n_<split>          : 세탁 엣지 수
    min_<split>        : 그 구간에서 가장 적은 클래스의 엣지 수 (클래스)
    unmatched_<split>% : 그 구간의 무패턴 비율
    cut_%              : 둘 이상 구간에 걸친 시도 비율
    test_with_earlier% : 테스트 패턴 엣지 중 같은 시도가 앞 구간(검증 포함)에도 있는 비율
    test_in_train%     : 테스트 패턴 엣지 중 같은 시도가 학습 구간에도 있는 비율 (라벨 누수 후보)
    test_new%          : 테스트 패턴 엣지 중 시도가 테스트에서 시작한 비율 (깨끗한 평가 묶음)
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
        lo = label_overlap(members, att, w)["edges"]
        if ("test", "ALL") in lo.index:
            r["test_in_train%"] = lo.loc[("test", "ALL"), "in_train_%"]
            r["test_new%"] = lo.loc[("test", "ALL"), "new_%"]
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


def plot_hub_phases(hub, patterns=("GATHER-SCATTER", "SCATTER-GATHER", "STACK", "CYCLE"), path=None):
    """hub(중간 계좌인 경우) 의 사이 간격 = 첫 출금 − 마지막 입금 (일) 분포. 0 오른쪽 = 다 받고 나서 보냄, 왼쪽 = 받기·보내기 겹침"""
    import matplotlib.pyplot as plt
    pats = [p for p in patterns if (hub["pattern_type"] == p).any()]
    fig, axes = plt.subplots(1, max(len(pats), 1), figsize=(4.5 * max(len(pats), 1), 3.6), squeeze=False)
    for ax, p in zip(axes[0], pats):
        g = hub.loc[hub["pattern_type"] == p, "gap_d"]
        ax.hist(g, bins=40, color="#1F77B4")
        ax.axvline(0, color="#D62728", lw=1, ls="--")
        ax.set_title(f"{p} hub: first_out - last_in (d)\nn={len(g):,}, >=0: {(g >= 0).mean()*100:.1f}%", fontsize=9)
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
