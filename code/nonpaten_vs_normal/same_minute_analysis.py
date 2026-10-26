"""03 notebook implementation; embedded in the notebook for standalone use."""
from pathlib import Path
from datetime import datetime
import time
import pandas as pd

BUCKETS = ['0', '1', '2~5', '6~20', '21+']
MINUTE_METRICS = [
    'pair_same_minute_other_n', 'receiver_same_minute_other_in_n',
    'receiver_same_minute_other_in_neighbors', 'receiver_same_minute_other_out_n',
]
METRICS = ['pair_prior_n', 'paid_currency_z', 'amt_z_pay_ccy',
           'receiver_in_gap_minutes', 'receiver_in_count'] + MINUTE_METRICS


def q(value):
    return "'" + str(value).replace("'", "''") + "'"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(con, title, sql):
    print(title, flush=True)
    started = time.perf_counter()
    con.execute(sql)
    print(f'  완료: {time.perf_counter()-started:,.1f}초', flush=True)


def bucket_sql(column):
    return (f"CASE WHEN {column}=0 THEN 0 WHEN {column}=1 THEN 1 "
            f"WHEN {column}<=5 THEN 2 WHEN {column}<=20 THEN 3 ELSE 4 END")


def extend_features(con, transactions, base_features, destination):
    """Preserve every base column; minute features use all labels, excluding self row.

    Both inputs must come from the same 02 run and original ledger. Validate target
    identity against the ledger before subtracting one from minute counts.
    """
    destination = Path(destination)
    require(not destination.exists(), f'기존 결과가 있습니다: {destination}')
    for p in [transactions, base_features]:
        require(Path(p).is_file(), f'입력 파일 없음: {p}')
    con.execute(f'CREATE OR REPLACE VIEW base AS SELECT * FROM read_parquet({q(base_features)})')
    con.execute(f'CREATE OR REPLACE VIEW ledger AS SELECT * FROM read_parquet({q(transactions)})')
    columns = {r[0] for r in con.execute('DESCRIBE base').fetchall()}
    required = {'edge_id','ts','src','dst','class_name','is_laundering','payment_currency',
                'payment_format','paid','pair_prior_n','paid_currency_z','amt_z_pay_ccy',
                'receiver_in_gap_minutes','receiver_in_count'}
    require(required <= columns, f'02 피처 컬럼 누락: {required-columns}')
    require(not (set(MINUTE_METRICS) | {'prior_bucket','prior_bucket_order'}) & columns,
            '이미 확장된 피처 대신 02번 features.parquet를 지정하세요.')
    n, unique_n, first, last, bad = con.execute("""
        SELECT count(*),count(DISTINCT edge_id),min(ts),max(ts),
          count(*) FILTER (WHERE ts IS NULL OR src IS NULL OR dst IS NULL
            OR ts IS DISTINCT FROM date_trunc('minute',ts)
            OR pair_prior_n IS NULL OR NOT isfinite(pair_prior_n)
            OR pair_prior_n<0 OR pair_prior_n<>floor(pair_prior_n)
            OR is_laundering IS NULL OR class_name IS DISTINCT FROM
              CASE WHEN is_laundering THEN 'nonpattern' ELSE 'normal' END)
        FROM base
    """).fetchone()
    require(n > 0 and n == unique_n and bad == 0, '02 피처의 ID·분 시각·이력 건수·라벨 오류')
    # Materialize only the target period; same-minute activity needs no older rows.
    # No class filter: known-pattern transactions remain in this table.
    run(con, '동일 분 집계용 전체 원장 준비 (패턴 거래 포함)', f"""
        CREATE OR REPLACE TABLE minute_source AS
        SELECT edge_id,ts,src,dst,paid,payment_currency,payment_format,is_laundering
        FROM ledger WHERE ts >= TIMESTAMP {q(first)} AND ts <= TIMESTAMP {q(last)}
    """)
    source_n, source_unique, source_bad = con.execute("""
        SELECT count(*),count(DISTINCT edge_id),count(*) FILTER (
          WHERE ts IS NULL OR src IS NULL OR dst IS NULL
            OR ts IS DISTINCT FROM date_trunc('minute',ts)) FROM minute_source
    """).fetchone()
    require(source_n == source_unique and source_bad == 0, '원장의 중복 ID·분 시각·계좌 오류')
    print('02 피처와 전체 원장의 거래 ID·값 일치 검사', flush=True)
    joined, mismatch = con.execute("""
        SELECT count(*),count(*) FILTER (WHERE t.edge_id IS NULL
          OR f.ts IS DISTINCT FROM t.ts OR f.src IS DISTINCT FROM t.src
          OR f.dst IS DISTINCT FROM t.dst OR f.paid IS DISTINCT FROM t.paid
          OR f.payment_currency IS DISTINCT FROM t.payment_currency
          OR f.payment_format IS DISTINCT FROM t.payment_format
          OR f.is_laundering IS DISTINCT FROM t.is_laundering)
        FROM base f LEFT JOIN minute_source t ON f.edge_id=t.edge_id
    """).fetchone()
    require(joined == n and mismatch == 0, '02 피처와 원장이 다른 실행 또는 다른 거래입니다.')
    run(con, '동일 분의 계좌쌍 및 수신 계좌 활동 집계', """
        CREATE OR REPLACE TABLE minute_pairs AS
        SELECT src,dst,ts,count(*)::BIGINT AS n
        FROM minute_source GROUP BY src,dst,ts;
        CREATE OR REPLACE TABLE minute_in AS
        SELECT dst,ts,sum(n)::BIGINT AS n,count(*)::BIGINT AS neighbors
        FROM minute_pairs GROUP BY dst,ts;
        CREATE OR REPLACE TABLE minute_out AS
        SELECT src,ts,sum(n)::BIGINT AS n FROM minute_pairs GROUP BY src,ts;
        DROP TABLE minute_source;
    """)
    run(con, '기존 과거 피처를 보존하고 동일 분 활동 저장', f"""
        COPY (
          SELECT f.*,{bucket_sql('f.pair_prior_n')} AS prior_bucket_order,
            CASE {bucket_sql('f.pair_prior_n')}
              WHEN 0 THEN '0' WHEN 1 THEN '1' WHEN 2 THEN '2~5'
              WHEN 3 THEN '6~20' ELSE '21+' END AS prior_bucket,
            p.n-1 AS pair_same_minute_other_n,
            i.n-1 AS receiver_same_minute_other_in_n,
            i.neighbors-CASE WHEN p.n=1 THEN 1 ELSE 0 END
              AS receiver_same_minute_other_in_neighbors,
            coalesce(o.n,0)-CASE WHEN f.src=f.dst THEN 1 ELSE 0 END
              AS receiver_same_minute_other_out_n
          FROM base f
          JOIN minute_pairs p ON f.src=p.src AND f.dst=p.dst AND f.ts=p.ts
          JOIN minute_in i ON f.dst=i.dst AND f.ts=i.ts
          LEFT JOIN minute_out o ON f.dst=o.src AND f.ts=o.ts
        ) TO {q(destination)} (FORMAT PARQUET,COMPRESSION ZSTD)
    """)
    con.execute(f'CREATE OR REPLACE VIEW extended AS SELECT * FROM read_parquet({q(destination)})')
    invalid = ' OR '.join(f'{m} IS NULL OR {m}<0' for m in MINUTE_METRICS)
    written, bad = con.execute(f"""
        SELECT count(*),count(*) FILTER (WHERE {invalid}
          OR pair_same_minute_other_n>receiver_same_minute_other_in_n
          OR receiver_same_minute_other_in_neighbors>receiver_same_minute_other_in_n)
        FROM extended
    """).fetchone()
    require(written == n and bad == 0, '추가 피처의 건수 보존·비음수 검사 실패')
    return dict(feature_rows=n, minute_source_rows=source_n,
                analysis_first=str(first), analysis_last=str(last),
                target_identity_mismatches=mismatch, invalid_feature_rows=bad)


def select_analysis_period(con, features, start, end_exclusive):
    """Filter target rows only; keep original historical/minute features unchanged."""
    start_ts, end_ts = map(datetime.fromisoformat, (start, end_exclusive))
    require(start_ts < end_ts, '분석 시작은 종료보다 빨라야 합니다.')
    require(all(t.second == 0 and t.microsecond == 0 for t in [start_ts, end_ts]),
            '동일 분 활동을 보존하도록 분석 경계를 분 단위로 지정하세요.')
    require(Path(features).is_file(), f'피처 파일 없음: {features}')
    con.execute(f"""
        CREATE OR REPLACE VIEW extended AS
        SELECT * FROM read_parquet({q(features)})
        WHERE ts >= TIMESTAMP {q(start)} AND ts < TIMESTAMP {q(end_exclusive)}
    """)
    n, normal, nonpattern, first, last = con.execute("""
        SELECT count(*),count(*) FILTER (WHERE class_name='normal'),
          count(*) FILTER (WHERE class_name='nonpattern'),min(ts),max(ts)
        FROM extended
    """).fetchone()
    require(n > 0 and normal > 0 and nonpattern > 0 and n == normal+nonpattern,
            '선택한 기간에 정상과 무패턴이 모두 있어야 하며 다른 클래스는 없어야 합니다.')
    return dict(analysis_start=start, analysis_end_exclusive=end_exclusive,
                analysis_rows=n, normal_rows=normal, nonpattern_rows=nonpattern,
                analysis_first=str(first), analysis_last=str(last))


def summarize(con, output):
    output = Path(output)
    tables = {}
    # Include empty prior bins. Empty denominators are NULL, never a zero fraud rate.
    tables['bucket_rates'] = con.execute("""
        WITH bins AS (SELECT * FROM (VALUES (0,'0'),(1,'1'),(2,'2~5'),
          (3,'6~20'),(4,'21+')) v(prior_bucket_order,prior_bucket)),
        counts AS (SELECT prior_bucket_order,
          count(*) FILTER (WHERE class_name='normal') AS normal_n,
          count(*) FILTER (WHERE class_name='nonpattern') AS nonpattern_n
          FROM extended GROUP BY 1)
        SELECT b.*,coalesce(normal_n,0) AS normal_n,coalesce(nonpattern_n,0) AS nonpattern_n,
          coalesce(normal_n,0)+coalesce(nonpattern_n,0) AS total_n,
          nonpattern_n::DOUBLE/nullif(normal_n+nonpattern_n,0) AS nonpattern_rate,
          normal_n::DOUBLE/nullif(sum(normal_n) OVER (),0) AS share_within_normal,
          nonpattern_n::DOUBLE/nullif(sum(nonpattern_n) OVER (),0) AS share_within_nonpattern
        FROM bins b LEFT JOIN counts USING(prior_bucket_order) ORDER BY 1
    """).df()
    for name, dims in [
        ('monthly_bucket_rates', "strftime(ts,'%Y-%m') AS month,prior_bucket_order,prior_bucket"),
        ('currency_format_bucket_rates', 'payment_currency,payment_format,prior_bucket_order,prior_bucket'),
    ]:
        tables[name] = con.execute(f"""
            SELECT {dims},count(*) AS total_n,
              count(*) FILTER (WHERE class_name='normal') AS normal_n,
              count(*) FILTER (WHERE class_name='nonpattern') AS nonpattern_n,
              avg((class_name='nonpattern')::INT) AS nonpattern_rate
            FROM extended GROUP BY ALL ORDER BY ALL
        """).df()
    tables['monthly_coverage'] = con.execute("""
        SELECT strftime(ts,'%Y-%m') AS month,class_name,count(*) AS n,
          min(ts) AS first_ts,max(ts) AS last_ts,
          count(DISTINCT CAST(ts AS DATE)) AS observed_days
        FROM extended GROUP BY 1,2 ORDER BY 1,2
    """).df()
    # Count-only cross tables let us compare same-minute activity within a prior bin.
    parts = []
    for m in MINUTE_METRICS:
        parts.append(f"""SELECT prior_bucket_order,prior_bucket,{q(m)} AS feature,
          {bucket_sql(m)} AS activity_bucket_order,
          count(*) AS total_n,count(*) FILTER (WHERE class_name='normal') AS normal_n,
          count(*) FILTER (WHERE class_name='nonpattern') AS nonpattern_n,
          avg((class_name='nonpattern')::INT) AS nonpattern_rate
          FROM extended GROUP BY 1,2,3,4""")
    tables['same_minute_rates'] = con.execute(' UNION ALL '.join(parts)+' ORDER BY 1,3,4').df()
    tables['same_minute_rates']['activity_bucket'] = tables['same_minute_rates'].activity_bucket_order.map(dict(enumerate(BUCKETS)))
    # Nulls remain missing; no-history gaps are not converted to zero.
    for name, dims, metrics in [
        ('feature_summary', ['prior_bucket_order','prior_bucket','class_name'], METRICS),
        ('monthly_feature_summary', ["strftime(ts,'%Y-%m') AS month",'prior_bucket_order','prior_bucket','class_name'], METRICS[:5]),
        ('currency_format_feature_summary', ['payment_currency','payment_format','prior_bucket_order','prior_bucket','class_name'], ['paid_currency_z','amt_z_pay_ccy','receiver_in_gap_minutes']),
    ]:
        print(f'분포 집계: {name}', flush=True)
        expressions = []
        for m in metrics:
            expressions += [f'count({m}) AS {m}__valid',f'avg({m}) AS {m}__mean',
                            f'count(*) FILTER (WHERE {m}=0) AS {m}__zeros',
                            f'approx_quantile({m},[.1,.25,.5,.75,.9,.99]) AS {m}__q']
        wide = con.execute('SELECT '+','.join(dims)+',count(*) AS n,'+','.join(expressions)+
                           ' FROM extended GROUP BY ALL ORDER BY ALL').df()
        records = []
        keys = [d.split(' AS ')[-1] for d in dims]
        for _, row in wide.iterrows():
            for m in metrics:
                valid = int(row[m+'__valid'])
                quantiles = row[m+'__q'] if valid else [None]*6
                records.append({**{k:row[k] for k in keys},'feature':m,'n':int(row.n),
                    'valid_n':valid,'missing_n':int(row.n)-valid,'mean':row[m+'__mean'],
                    'zero_rate_valid':row[m+'__zeros']/valid if valid else None,
                    **dict(zip(['p10','p25','p50','p75','p90','p99'],quantiles))})
        tables[name] = pd.DataFrame(records)
    for name, df in tables.items():
        df.to_csv(output/f'{name}.csv',index=False,encoding='utf-8-sig')
    return tables


def plot_results(tables, output):
    import numpy as np
    import matplotlib.pyplot as plt
    output = Path(output)
    df = tables['bucket_rates'].set_index('prior_bucket').reindex(BUCKETS)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(BUCKETS,df.nonpattern_rate*100)
    ax.set(xlabel='Prior same-direction pair count',ylabel='Nonpattern / (normal + nonpattern), %')
    fig.tight_layout();fig.savefig(output/'bucket_nonpattern_rate.png',dpi=150)
    plt.show();plt.close(fig)
    stats = tables['feature_summary']
    for filename, metrics in [('amount_gap_by_bucket',['paid_currency_z','amt_z_pay_ccy','receiver_in_gap_minutes']),
                              ('same_minute_by_bucket',MINUTE_METRICS)]:
        fig, axes = plt.subplots(1,len(metrics),figsize=(5*len(metrics),4),squeeze=False)
        for ax,m in zip(axes.flat,metrics):
            for offset,cl in [(-.12,'normal'),(.12,'nonpattern')]:
                part = stats[(stats.feature==m)&(stats.class_name==cl)].set_index('prior_bucket').reindex(BUCKETS)
                med = part.p50.to_numpy(float)
                ax.errorbar(np.arange(5)+offset,med,
                    yerr=[np.maximum(0,med-part.p25.to_numpy(float)),np.maximum(0,part.p75.to_numpy(float)-med)],
                    fmt='o',capsize=3,label=cl)
            ax.set_xticks(range(5),BUCKETS);ax.set_title(m,fontsize=9);ax.legend()
            ax.set_xlabel('Prior pair count');ax.set_ylabel('Median and p25-p75 (approx.)')
            if m not in ['paid_currency_z','amt_z_pay_ccy']:ax.set_yscale('symlog',linthresh=1)
        fig.tight_layout();fig.savefig(output/f'{filename}.png',dpi=150)
        plt.show();plt.close(fig)
