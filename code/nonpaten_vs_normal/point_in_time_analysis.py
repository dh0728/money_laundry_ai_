"""02 노트북용 DuckDB 분석. 이력 계산에는 라벨·패턴 정보를 사용하지 않는다."""
from pathlib import Path
from datetime import datetime
import time


METRICS = [
    "paid_currency_z", "amt_z_pay_ccy", "pair_prior_n",
    "receiver_in_count", "receiver_out_count", "receiver_in_neighbors",
    "receiver_out_neighbors", "receiver_active_days", "receiver_in_count_1h",
    "receiver_out_count_1h", "receiver_in_gap_minutes", "receiver_in_per_active_day",
]


def q(value):
    return "'" + str(value).replace("'", "''") + "'"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(con, title, sql):
    print(title, flush=True)
    start = time.perf_counter()
    con.execute(sql)
    print(f"  완료: {time.perf_counter() - start:,.1f}초", flush=True)


def prepare_inputs(con, transactions, patterns, reference_start, reference_end,
                   analysis_start, analysis_end=None):
    """모든 종료 시각은 미포함. 분석 하한으로 이력 원장을 자르지 않는다."""
    r0, r1, a0 = map(datetime.fromisoformat, (reference_start, reference_end, analysis_start))
    require(r0 < r1 <= a0, "금액 기준 기간 종료는 분석 시작 이하여야 합니다.")
    if analysis_end is not None:
        require(a0 < datetime.fromisoformat(analysis_end), "분석 종료는 시작보다 늦어야 합니다.")
    for path in (transactions, patterns):
        require(Path(path).is_file(), f"입력 파일이 없습니다: {path}")
    con.execute(f"CREATE OR REPLACE VIEW source_tx AS SELECT * FROM read_parquet({q(transactions)})")
    schema = dict((r[0], r[1]) for r in con.execute("DESCRIBE source_tx").fetchall())
    required = {"edge_id", "ts", "src", "dst", "paid", "payment_currency", "payment_format", "is_laundering"}
    require(required <= schema.keys(), f"필수 거래 컬럼 누락: {required - schema.keys()}")
    require(schema["is_laundering"] == "BOOLEAN", "01 전처리의 BOOLEAN 라벨을 사용하세요.")
    con.execute(f"CREATE OR REPLACE TABLE pattern_ids AS SELECT edge_id, ts, unmatched FROM read_parquet({q(patterns)})")
    n, unique_n, bad = con.execute("""
        SELECT count(*), count(DISTINCT edge_id),
          count(*) FILTER (WHERE edge_id IS NULL OR ts IS NULL OR unmatched IS DISTINCT FROM FALSE)
        FROM pattern_ids
    """).fetchone()
    require(n > 0 and n == unique_n and bad == 0, "패턴 파일이 비었거나 중복 ID·미매칭 행이 있습니다.")
    joined, invalid = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE t.edge_id IS NULL
          OR t.ts IS DISTINCT FROM p.ts OR t.is_laundering IS DISTINCT FROM TRUE)
        FROM pattern_ids p LEFT JOIN source_tx t ON t.edge_id=p.edge_id
    """).fetchone()
    require(joined == n and invalid == 0, "패턴과 거래의 ID·시각·양성 라벨이 일치하지 않습니다.")
    upper = f"WHERE ts < TIMESTAMP {q(analysis_end)}" if analysis_end else ""
    run(con, "전체 이력 원장 준비 (패턴 거래 포함)", f"""
        CREATE OR REPLACE TABLE history AS
        SELECT edge_id, ts, src, dst, paid, payment_currency, payment_format, is_laundering
        FROM source_tx {upper}
    """)
    total, invalid, first, last = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE edge_id IS NULL OR ts IS NULL
          OR src IS NULL OR dst IS NULL OR is_laundering IS NULL
          OR paid IS NULL OR NOT isfinite(paid) OR paid <= 0
          OR payment_currency IS NULL OR payment_format IS NULL
          OR ts IS DISTINCT FROM date_trunc('minute', ts)), min(ts), max(ts)
        FROM history
    """).fetchone()
    require(total > 0 and invalid == 0, f"원장 필수값·금액·분 단위 시각 오류: {invalid:,}")
    # 패턴은 비교 대상 선정에만 사용한다. 아래 이력 집계는 history 전체를 읽는다.
    con.execute(f"""
        CREATE OR REPLACE TABLE targets AS
        SELECT h.* FROM history h ANTI JOIN pattern_ids p ON h.edge_id=p.edge_id
        WHERE h.ts >= TIMESTAMP {q(analysis_start)}
    """)
    counts = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE NOT is_laundering),
          count(*) FILTER (WHERE is_laundering) FROM targets
    """).fetchone()
    require(counts[1] > 0 and counts[2] > 0, "분석 기간에 정상과 무패턴 자금세탁이 모두 있어야 합니다.")
    return {"history_rows": total, "history_start": str(first), "history_end": str(last),
            "all_pattern_rows": n, "target_rows": counts[0], "normal_rows": counts[1],
            "nonpattern_rows": counts[2], "reference_start": reference_start,
            "reference_end_exclusive": reference_end, "analysis_start": analysis_start,
            "analysis_end_exclusive": analysis_end}


def fit_amount_reference(con, start, end):
    """고정된 분석 이전 기간만 사용. V10 패키지 보정표를 재사용하지 않는다."""
    run(con, "분석 이전 기간의 통화별 금액 보정값 계산", f"""
        CREATE OR REPLACE TABLE reference_amounts AS
        SELECT paid, payment_currency FROM history
        WHERE ts >= TIMESTAMP {q(start)} AND ts < TIMESTAMP {q(end)};
        CREATE OR REPLACE TABLE amount_scale AS
        SELECT payment_currency, count(*) AS reference_n, median(paid) AS scale
        FROM reference_amounts GROUP BY payment_currency;
        CREATE OR REPLACE TABLE amount_reference AS
        SELECT payment_currency, any_value(reference_n) AS reference_n,
          any_value(scale) AS scale, median(ln(1 + paid/scale)) AS med,
          quantile_cont(ln(1 + paid/scale), .75)-quantile_cont(ln(1 + paid/scale), .25) AS raw_iqr,
          greatest(quantile_cont(ln(1 + paid/scale), .75)-quantile_cont(ln(1 + paid/scale), .25), 1e-6) AS iqr,
          avg(ln(1+paid)) AS log_mean, stddev_samp(ln(1+paid)) AS log_std
        FROM reference_amounts JOIN amount_scale USING (payment_currency)
        GROUP BY payment_currency;
        DROP TABLE reference_amounts;
        DROP TABLE amount_scale;
    """)
    require(con.execute("SELECT count(*) FROM amount_reference").fetchone()[0] > 0,
            "금액 보정 기준 기간에 거래가 없습니다.")


def build_history_states(con):
    """시각 t의 상태에는 t까지 포함된다. 조회 시 strict ASOF (< t)로 현재 분을 뺀다."""
    run(con, "계좌쌍의 분별 누적 거래 수", """
        CREATE OR REPLACE TABLE pair_state AS
        SELECT src, dst, ts,
          sum(n) OVER (PARTITION BY src,dst ORDER BY ts ROWS UNBOUNDED PRECEDING)::BIGINT AS prior_n
        FROM (SELECT src,dst,ts,count(*) AS n FROM history GROUP BY src,dst,ts);
        CREATE OR REPLACE TABLE first_pairs AS
        SELECT src,dst,min(ts) AS ts FROM history GROUP BY src,dst;
    """)
    # 일별 최초 활동도 사건으로 만든다. 해당 일의 첫 거래와 같은 분에는 아직 활동일에 더하지 않는다.
    run(con, "계좌별 활동 사건 집계", """
        CREATE OR REPLACE TABLE node_events AS
        WITH events AS (
          SELECT dst AS node,ts,count(*) AS in_n,0::BIGINT AS out_n,
            0::BIGINT AS new_in,0::BIGINT AS new_out,0::BIGINT AS new_day
          FROM history GROUP BY dst,ts
          UNION ALL
          SELECT src,ts,0,count(*),0,0,0 FROM history GROUP BY src,ts
          UNION ALL
          SELECT dst,ts,0,0,count(*),0,0 FROM first_pairs GROUP BY dst,ts
          UNION ALL
          SELECT src,ts,0,0,0,count(*),0 FROM first_pairs GROUP BY src,ts
          UNION ALL
          SELECT node,min(ts),0,0,0,0,1 FROM (
            SELECT src AS node,ts FROM history UNION ALL SELECT dst,ts FROM history
          ) GROUP BY node,CAST(ts AS DATE)
        )
        SELECT node,ts,sum(in_n) AS in_n,sum(out_n) AS out_n,
          sum(new_in) AS new_in,sum(new_out) AS new_out,sum(new_day) AS new_day
        FROM events GROUP BY node,ts;
        DROP TABLE first_pairs;
    """)
    run(con, "수신 계좌 조회용 누적 상태 계산", """
        CREATE OR REPLACE TABLE node_state AS
        SELECT node,ts,
          (sum(in_n) OVER w)::BIGINT AS in_count, (sum(out_n) OVER w)::BIGINT AS out_count,
          (sum(new_in) OVER w)::BIGINT AS in_neighbors, (sum(new_out) OVER w)::BIGINT AS out_neighbors,
          (sum(new_day) OVER w)::BIGINT AS active_days,
          max(CASE WHEN in_n>0 THEN ts END) OVER w AS last_in_ts
        FROM node_events WINDOW w AS (PARTITION BY node ORDER BY ts ROWS UNBOUNDED PRECEDING);
        DROP TABLE node_events;
    """)


def write_features(con, path, expected_rows):
    path = Path(path)
    require(not path.exists(), f"기존 결과를 덮어쓰지 않습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # 범위 [t-60분,t): 직전 누적값에서 t-60분 미만의 누적값을 뺀다.
    run(con, "거래 직전 피처 결합 및 Parquet 저장", f"""
        COPY (
          WITH joined AS (
            SELECT t.edge_id,t.ts,t.src,t.dst,t.paid,t.payment_currency,t.payment_format,t.is_laundering,
              CASE WHEN t.is_laundering THEN 'nonpattern' ELSE 'normal' END AS class_name,
              coalesce(p.prior_n,0) AS pair_prior_n,
              coalesce(s.in_count,0) AS receiver_in_count,
              coalesce(s.out_count,0) AS receiver_out_count,
              coalesce(s.in_neighbors,0) AS receiver_in_neighbors,
              coalesce(s.out_neighbors,0) AS receiver_out_neighbors,
              coalesce(s.active_days,0) AS receiver_active_days,
              coalesce(s.in_count,0)-coalesce(old.in_count,0) AS receiver_in_count_1h,
              coalesce(s.out_count,0)-coalesce(old.out_count,0) AS receiver_out_count_1h,
              date_diff('minute',s.last_in_ts,t.ts) AS receiver_in_gap_minutes,
              s.in_count::DOUBLE / nullif(s.active_days,0) AS receiver_in_per_active_day,
              CASE WHEN a.scale IS NOT NULL THEN
                greatest(-8.0,least(8.0,(ln(1+t.paid/a.scale)-a.med)/a.iqr)) END AS paid_currency_z,
              CASE WHEN a.log_std>0 THEN
                greatest(-10.0,least(10.0,(ln(1+t.paid)-a.log_mean)/(a.log_std+1e-6))) END AS amt_z_pay_ccy
            FROM targets t
            ASOF LEFT JOIN pair_state p ON t.src=p.src AND t.dst=p.dst AND t.ts>p.ts
            ASOF LEFT JOIN node_state s ON t.dst=s.node AND t.ts>s.ts
            ASOF LEFT JOIN node_state old ON t.dst=old.node AND t.ts-INTERVAL '60 minutes'>old.ts
            LEFT JOIN amount_reference a ON t.payment_currency=a.payment_currency
          )
          SELECT *,ln(1+pair_prior_n) AS pair_prior_count,
            CASE WHEN pair_prior_n=0 THEN 'zero' ELSE 'one_or_more' END AS pair_group
          FROM joined
        ) TO {q(path)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"CREATE OR REPLACE VIEW features AS SELECT * FROM read_parquet({q(path)})")
    invalid_expr = " OR ".join(f"({m} IS NOT NULL AND NOT isfinite({m}))" for m in METRICS)
    n, bad = con.execute(f"""
        SELECT count(*),count(*) FILTER (WHERE {invalid_expr}
          OR pair_prior_n<0 OR receiver_in_count<0 OR receiver_out_count<0
          OR receiver_in_count_1h<0 OR receiver_out_count_1h<0
          OR receiver_in_count_1h>receiver_in_count OR receiver_out_count_1h>receiver_out_count
          OR receiver_in_neighbors>receiver_in_count OR receiver_out_neighbors>receiver_out_count)
        FROM features
    """).fetchone()
    require(n == expected_rows and bad == 0, f"피처 검증 실패: {n:,}행, 오류 {bad:,}행")
    actual = con.execute("SELECT is_laundering,count(*) FROM features GROUP BY 1 ORDER BY 1").fetchall()
    expected = con.execute("SELECT is_laundering,count(*) FROM targets GROUP BY 1 ORDER BY 1").fetchall()
    require(actual == expected, "라벨별 피처 행 수가 비교 대상과 다릅니다.")
    return {"feature_rows": n, "invalid_feature_rows": bad,
            "robust_z_missing": con.execute("SELECT count(*) FROM features WHERE paid_currency_z IS NULL").fetchone()[0]}


def summarize(con, output_dir):
    """원본 전량은 DuckDB에서 집계하고 작은 집계표만 pandas로 반환한다."""
    import pandas as pd
    output_dir = Path(output_dir)
    queries = {
        "group_counts": """
          SELECT class_name,pair_group,count(*) AS n,
            count(*)::DOUBLE/sum(count(*)) OVER (PARTITION BY class_name) AS share_within_class,
            count(*)::DOUBLE/sum(count(*)) OVER (PARTITION BY pair_group) AS share_within_pair_group
          FROM features GROUP BY 1,2 ORDER BY 1,2
        """,
        "monthly_counts": """
          SELECT strftime(ts,'%Y-%m') AS month,class_name,pair_group,count(*) AS n,
            count(*)::DOUBLE/sum(count(*)) OVER (PARTITION BY strftime(ts,'%Y-%m'),class_name) AS share_within_class
          FROM features GROUP BY 1,2,3 ORDER BY 1,2,3
        """,
        "currency_format_counts": """
          SELECT payment_currency,payment_format,pair_group,class_name,count(*) AS n,
            count(*)::DOUBLE/sum(count(*)) OVER (PARTITION BY payment_currency,payment_format,pair_group) AS share_within_stratum
          FROM features GROUP BY 1,2,3,4 ORDER BY 1,2,3,4
        """,
        "z_histogram": """
          SELECT class_name,pair_group,
            CASE WHEN paid_currency_z IS NULL THEN NULL
              ELSE least(31,floor((paid_currency_z+8)/.5)::INTEGER) END AS bin,
            count(*) AS n FROM features GROUP BY 1,2,3 ORDER BY 1,2,3
        """,
    }
    tables = {}
    for name, sql in queries.items():
        print(f"집계: {name}", flush=True)
        tables[name] = con.execute(sql).df()
    # 여러 피처의 분위수를 한 번의 스캔으로 집계한다.
    expressions = []
    for m in METRICS:
        expressions += [f"count({m}) AS {m}__valid", f"avg({m}) AS {m}__mean",
                        f"count(*) FILTER (WHERE {m}=0) AS {m}__zeros",
                        f"approx_quantile({m},[.10,.25,.50,.75,.90,.99]) AS {m}__q"]
    print("집계: 피처별 분포 (분위수는 근사값)", flush=True)
    wide = con.execute("SELECT class_name,pair_group,count(*) AS n," + ",".join(expressions) +
                       " FROM features GROUP BY 1,2 ORDER BY 1,2").df()
    records = []
    for _, row in wide.iterrows():
        for m in METRICS:
            valid = int(row[m+"__valid"])
            quantiles = row[m+"__q"] if valid else [None]*6
            records.append({"class_name": row.class_name, "pair_group": row.pair_group,
                "feature": m, "n": int(row.n), "valid_n": valid, "missing_n": int(row.n)-valid,
                "mean": row[m+"__mean"], "zero_rate_valid": row[m+"__zeros"]/valid if valid else None,
                **dict(zip(["p10","p25","p50","p75","p90","p99"],quantiles))})
    tables["feature_summary"] = pd.DataFrame(records)
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    return tables


def plot_results(tables, output_dir):
    import numpy as np
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    groups = [(c,p) for c in ("normal","nonpattern") for p in ("zero","one_or_more")]
    labels = [f"{c} / prior {'0' if p=='zero' else '>=1'}" for c,p in groups]
    counts = tables["group_counts"].pivot(index="class_name",columns="pair_group",values="share_within_class")
    counts = counts.reindex(index=["normal","nonpattern"],columns=["zero","one_or_more"]).fillna(0)
    ax = (counts*100).plot.bar(rot=0,figsize=(7,4),ylabel="Within-class share (%)",title="Prior same-direction pair history")
    ax.figure.tight_layout(); ax.figure.savefig(output_dir/"pair_group_share.png",dpi=150)
    plt.show(); plt.close(ax.figure)
    features = ["paid_currency_z","receiver_in_count","receiver_out_count",
                "receiver_in_neighbors","receiver_active_days","receiver_in_count_1h"]
    fig, axes = plt.subplots(2,3,figsize=(16,8))
    for ax, name in zip(axes.flat,features):
        df = tables["feature_summary"].query("feature == @name").set_index(["class_name","pair_group"]).reindex(groups)
        med = df.p50.to_numpy(dtype=float)
        lo, hi = df.p10.to_numpy(dtype=float), df.p90.to_numpy(dtype=float)
        ax.errorbar(med,np.arange(4),xerr=[np.maximum(0,med-lo),np.maximum(0,hi-med)],fmt='o',capsize=4)
        ax.set_yticks(range(4),labels); ax.invert_yaxis(); ax.set_title(name)
        if name != "paid_currency_z": ax.set_xscale('symlog',linthresh=1)
        ax.set_xlabel("Median and p10-p90 (approximate)"); ax.grid(axis='x',alpha=.25)
    fig.tight_layout(); fig.savefig(output_dir/"feature_intervals.png",dpi=150)
    plt.show(); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4),sharey=True)
    hist = tables['z_histogram']
    for ax,pair_group in zip(axes,['zero','one_or_more']):
        for class_name in ['normal','nonpattern']:
            group = hist[(hist.pair_group==pair_group)&(hist.class_name==class_name)&hist.bin.notna()]
            bins = group.set_index('bin').n.reindex(range(32),fill_value=0)
            if bins.sum(): ax.step(-8+(np.arange(32)+.5)*.5,bins/bins.sum(),where='mid',label=class_name)
        ax.set_title(f"pair prior: {pair_group}"); ax.set_xlabel('paid_currency_z'); ax.legend()
    axes[0].set_ylabel('Within-group fraction (nonmissing z only)')
    fig.tight_layout(); fig.savefig(output_dir/'amount_z_distribution.png',dpi=150)
    plt.show(); plt.close(fig)
