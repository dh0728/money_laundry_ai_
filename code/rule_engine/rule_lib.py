"""패턴 룰 실험 공통 도구 (03_rule_vs_model).

모델 점수를 전혀 쓰지 않고 원장(거래 표)에 패턴 룰 6종을 직접 돌린다. 계산은 DuckDB 안에서 한다.

룰 하나는 조건 여러 개를 AND 로 묶은 것이다.
  ① 거래 필터    : 룰이 세는 거래를 먼저 거른다 (건당 최소 금액, 처음 거래하는 상대만). apply_filter()
  ② 기간         : 트리거 시각을 포함한 직전 W 분. 원장이 분 단위라 [t0 - (W-1)분, t0]
  ③ 세기 ≥ 조건값 : 기간 안에서 센 패턴의 크기
  ④ (FAN 만) 반복 없음: 기간 안 거래 수 = 상대 계좌 수. 같은 상대와 두 번 이상 거래하지 않았다는 뜻

룰마다 "거래별 세기" 표 <룰>_<W>_<필터 tag> 를 만든다. 세기가 조건값 이상인 거래가 그 조건에서 룰이 고른 거래(근거 거래)다.

  표                   v(세기)의 뜻
  fan_out_*            그 거래를 담는 기간들 중, 서로 다른 수취 계좌 수의 최댓값 (v_once: 반복 없음 조건을 건 값)
  fan_in_*             서로 다른 송금 계좌 수의 최댓값 (v_once 동일)
  gather_scatter_*     min(송금 계좌 수, 첫 입금 뒤 수취 계좌 수)의 최댓값
  scatter_gather_*     경유 계좌 수의 최댓값
  cycle_*              그 거래가 든 순환 중 가장 긴 것의 거래 수
  random_*             그 거래가 든 (돌아오지 않는) 경로 중 가장 긴 것의 거래 수

세기를 저장해 두므로 조건값을 바꿀 때 룰을 다시 돌리지 않고 걸러 보기만 하면 된다.

공통 규칙
  - 자기 자신에게 보낸 거래(src = dst)는 룰이 보지 않는다.
  - 바쁜 계좌(busy 표)는 대상도 경유도 되지 않는다. 상대 계좌로는 센다.
  - 금액은 보낸 금액을 서비스 고정 환율로 바꾼 USD 다 (gen_lib.UNITS_PER_USD).
  - 지표에서 빼는 세탁(load_tx 의 exclude, 열 skip)은 **정답도 오답도 아닌 것으로 두고 뺀다.**
    룰은 원장 전체를 그대로 보지만, 이 거래는 고른 거래 수에도 세탁 수에도 분모에도 들어가지 않는다.

검증: HI-Small 에서 여섯 룰의 SQL 결과가 같은 정의를 Python 반복문으로 그대로 짠 계산과 일치하는 것을 확인했다(2026-11-23).
      FAN(반복 없음 포함)·GATHER-SCATTER 는 무작위 계좌 400개씩, SCATTER-GATHER 와 경로형은 전체.
"""
import time

import numpy as np
import pandas as pd

RULES = {'fan_out': 'FAN-OUT', 'fan_in': 'FAN-IN', 'gather_scatter': 'GATHER-SCATTER',      # 룰 이름 → Patterns 파일의 패턴 이름
         'scatter_gather': 'SCATTER-GATHER', 'cycle': 'CYCLE', 'random': 'RANDOM'}
REPEAT = {'허용': 'v', '없음': 'v_once'}          # FAN 의 '반복' 조건 → 세기 열
KEYS = ['룰', '필터', '반복', '기간_분', '조건값']   # 조건 한 벌을 가리키는 열


def _timed(con, sql):
    started = time.perf_counter()
    con.execute(sql)
    return round(time.perf_counter() - started, 1)


def _rows(con, table):
    return con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]


def _between(window):
    return f"BETWEEN DATE '{window[0]}' AND DATE '{window[1]}'"


# ----------------------------------------------------------------------------- 거래 올리기
def load_tx(con, base, start, end, units_per_usd, exclude=()):
    """[start, end] 날짜의 거래를 DuckDB 표로 올린다.

    tx_all : 읽은 거래 전부. 지표는 여기서 센다
    tx     : 룰이 보는 거래. 자기 자신에게 보낸 거래를 뺀 것
    exclude: 지표에서 빼는 세탁의 pattern_type 목록 (예: ['NO_PATTERN', 'STACK', 'BIPARTITE'])
    열     : edge_id, ts, day, src, dst, usd, y, pattern_type, attempt_id, first_pair, skip
             pattern_type 은 정상이면 '', 패턴 파일에 없는 세탁이면 'NO_PATTERN'
             first_pair 는 그 (송금 계좌, 수취 계좌) 쌍의 첫 거래인가. 읽는 기간이 아니라 **데이터 처음부터 end 까지** 전체에서 본다
             skip 은 지표에서 빼는 거래인가 (세탁이고 pattern_type 이 exclude 에 있다). 룰은 이 거래도 그대로 본다
    """
    skipped = ', '.join("'" + kind.replace("'", "''") + "'" for kind in exclude) or "'(없음)'"
    rates = ', '.join("('" + currency.replace("'", "''") + f"', {units!r})" for currency, units in units_per_usd.items())
    con.execute(f'CREATE OR REPLACE TEMP TABLE fx_rates AS SELECT * FROM (VALUES {rates}) AS r(currency, units)')
    source = f"read_parquet('{base / 'transactions.parquet'}')"
    con.execute(f"""
        CREATE OR REPLACE TABLE tx_all AS
        WITH pat AS (SELECT edge_id, min(attempt_id) AS attempt_id, arg_min(pattern_type, attempt_id) AS pattern_type
                     FROM read_parquet('{base / 'patterns.parquet'}') WHERE edge_id IS NOT NULL GROUP BY edge_id),
        pair AS (SELECT min(edge_id) AS edge_id FROM {source}        -- edge_id 는 시간순이라 최솟값이 그 쌍의 첫 거래다
                 WHERE day <= DATE '{end}' AND src <> dst GROUP BY src, dst)
        SELECT t.edge_id, t.ts, t.day, t.src, t.dst, round(t.paid / f.units, 6) AS usd,
               CAST(t.is_laundering AS TINYINT) AS y,
               CASE WHEN t.is_laundering THEN coalesce(p.pattern_type, 'NO_PATTERN') ELSE '' END AS pattern_type,
               CASE WHEN t.is_laundering THEN p.attempt_id END AS attempt_id,
               pair.edge_id IS NOT NULL AS first_pair,
               t.is_laundering AND coalesce(p.pattern_type, 'NO_PATTERN') IN ({skipped}) AS skip
        FROM {source} t
        LEFT JOIN fx_rates f ON f.currency = t.payment_currency
        LEFT JOIN pat p USING (edge_id)
        LEFT JOIN pair USING (edge_id)
        WHERE t.day BETWEEN DATE '{start}' AND DATE '{end}'""")
    con.execute('CREATE OR REPLACE TABLE tx AS SELECT * FROM tx_all WHERE src <> dst')
    n, first, last, unknown, skip = con.execute(
        'SELECT count(*), min(ts), max(ts), count(*) FILTER (WHERE usd IS NULL), count(*) FILTER (WHERE skip) FROM tx_all').fetchone()
    return dict(읽은_거래=n, 룰이_보는_거래=_rows(con, 'tx'), 시작=str(first), 끝=str(last), 환율_모르는_거래=unknown, 지표에서_뺀_세탁=skip)


def busy_accounts(con, busy_max, busy_minutes):
    """바쁜 계좌: 어느 busy_minutes 구간에서든 보낸 거래 또는 받은 거래가 busy_max 건을 넘는 계좌.
    거래 필터를 걸기 전의 거래(tx)로 센다. busy_max 가 None 이면 바쁜 계좌를 두지 않는다."""
    if busy_max is None:
        con.execute('CREATE OR REPLACE TABLE busy AS SELECT src AS a FROM tx WHERE false')
    else:
        frame = f'RANGE BETWEEN INTERVAL {busy_minutes - 1} MINUTES PRECEDING AND CURRENT ROW'
        con.execute(f"""
            CREATE OR REPLACE TABLE busy AS
            SELECT DISTINCT a FROM (
                SELECT src AS a, count(*) OVER (PARTITION BY src ORDER BY ts {frame}) AS n FROM tx
                UNION ALL
                SELECT dst, count(*) OVER (PARTITION BY dst ORDER BY ts {frame}) FROM tx)
            WHERE n > {busy_max}""")
    touched, laundering = con.execute("""SELECT count(*), coalesce(sum(y), 0) FROM tx
                                         WHERE src IN (SELECT a FROM busy) OR dst IN (SELECT a FROM busy)""").fetchone()
    return dict(바쁜_계좌=_rows(con, 'busy'), 바쁜_계좌에_닿은_거래=touched, 그중_세탁=int(laundering))


def apply_filter(con, tag, min_usd, first_only):
    """거래 필터 하나를 건다. 룰은 이 필터를 통과한 거래만 세고, 근거 거래도 그 안에서만 나온다.

    min_usd    : 건당 최소 금액(USD). 이보다 작은 거래는 세지 않는다
    first_only : True 면 그 (송금 계좌, 수취 계좌) 쌍의 첫 거래만 센다 = 처음 거래하는 상대
    만드는 표  : tx_<tag> (필터를 통과한 거래), txn_<tag> (그중 양쪽 어디에도 바쁜 계좌가 없는 거래)
    """
    where = f'usd >= {min_usd}' + (' AND first_pair' if first_only else '')
    con.execute(f'CREATE OR REPLACE TABLE tx_{tag} AS SELECT * FROM tx WHERE {where}')
    con.execute(f"""CREATE OR REPLACE TABLE txn_{tag} AS
                    SELECT * FROM tx_{tag} WHERE src NOT IN (SELECT a FROM busy) AND dst NOT IN (SELECT a FROM busy)""")
    n, laundering, patterned = con.execute(f"""SELECT count(*), coalesce(sum(y), 0),
        count(*) FILTER (WHERE pattern_type IN ({', '.join(repr(v) for v in RULES.values())})) FROM tx_{tag}""").fetchone()
    return dict(tag=tag, 건당_최소_USD=min_usd, 첫_거래만=first_only, 남은_거래=n, 그중_세탁=int(laundering), 그중_6종_패턴_세탁=patterned)


# ----------------------------------------------------------------------------- ①② FAN-OUT · FAN-IN (집계형)
def fan(con, side, W, tag):
    """FAN-OUT(side='out') / FAN-IN(side='in').

    대상 계좌 A 의 거래 하나하나가 트리거다. 트리거 시각에서 기간 W 만큼 돌아봐 서로 다른 상대 계좌 수 k 와 거래 수 n 을 센다.
    k >= K 인 트리거의 기간 안에 든 A 의 같은 방향 거래 전부가 근거 거래다.
      v      = 그 거래를 기간 안에 담는 트리거들의 k 중 최댓값
      v_once = 같은 값인데 "반복 없음"(n = k: 기간 안에서 모든 상대와 한 번씩만 거래)인 트리거만 본 것
    따라서 v >= K (또는 v_once >= K) 인 거래가 조건값 K 의 근거 거래다.
    """
    subject, other = ('src', 'dst') if side == 'out' else ('dst', 'src')
    table = f'fan_{side}_{W}_{tag}'
    back = f'INTERVAL {W - 1} MINUTES'
    seconds = _timed(con, f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH trig AS (                                   -- 트리거마다: 기간 안 서로 다른 상대 계좌 수와 거래 수
            SELECT edge_id, {subject} AS a, ts, count(DISTINCT {other}) OVER w AS k, count(*) OVER w AS n
            FROM tx_{tag} WHERE {subject} NOT IN (SELECT a FROM busy)
            WINDOW w AS (PARTITION BY {subject} ORDER BY ts RANGE BETWEEN {back} PRECEDING AND CURRENT ROW)
        ), reach AS (                                    -- 거래마다: 자기를 담는 트리거들의 k 최댓값
            SELECT edge_id, a AS subj, max(k) OVER fwd AS v, coalesce(max(CASE WHEN n = k THEN k END) OVER fwd, 0) AS v_once
            FROM trig
            WINDOW fwd AS (PARTITION BY a ORDER BY ts RANGE BETWEEN CURRENT ROW AND {back} FOLLOWING))
        SELECT edge_id, v, v_once, subj FROM reach WHERE v >= 2""")
    return dict(표=table, 거래=_rows(con, table), 초=seconds)


# ----------------------------------------------------------------------------- ③ GATHER-SCATTER (집계형)
def gather_scatter(con, W, tag):
    """GATHER-SCATTER: 가운데 계좌 H 가 여러 계좌에서 받은 뒤 여러 계좌로 보낸다.

    H 의 출금 하나하나가 트리거다. 트리거 시각 t0 에서 기간 W 만큼 돌아봐
      din  = 기간 안 입금의 서로 다른 송금 계좌 수
      dout = 기간 안 첫 입금 시각부터 t0 까지 출금의 서로 다른 수취 계좌 수   ("모은 뒤에 뿌린다")
    를 세고, 트리거의 세기를 g = min(din, dout) 로 둔다. g >= K 면 적중이다.
    근거 거래 = 기간 안 입금 전부 + 첫 입금 뒤의 출금.
    거래별 세기 v = 그 거래를 근거로 삼는 트리거들의 g 중 최댓값.
    """
    table = f'gather_scatter_{W}_{tag}'
    back = f'INTERVAL {W - 1} MINUTES'
    seconds = _timed(con, f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH ev AS (                                     -- 계좌 쪽에서 본 거래. dir 0 = 받음, 1 = 보냄
            SELECT edge_id, dst AS a, src AS cp, ts, 0 AS dir FROM tx_{tag} WHERE dst NOT IN (SELECT a FROM busy)
            UNION ALL
            SELECT edge_id, src, dst, ts, 1 FROM tx_{tag} WHERE src NOT IN (SELECT a FROM busy)
        ), s1 AS (
            SELECT *,
                   count(DISTINCT CASE WHEN dir = 0 THEN cp END) OVER w AS din,
                   min(CASE WHEN dir = 0 THEN ts END) OVER w AS first_in,
                   max(CASE WHEN dir = 0 THEN ts END) OVER (PARTITION BY a ORDER BY ts
                         RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_in      -- 이 시각까지의 마지막 입금
            FROM ev
            WINDOW w AS (PARTITION BY a ORDER BY ts RANGE BETWEEN {back} PRECEDING AND CURRENT ROW)
        ), s2 AS (                                       -- 출금 트리거의 세기 g
            SELECT *,
                   CASE WHEN dir = 1 AND first_in IS NOT NULL THEN least(din,
                        count(DISTINCT CASE WHEN dir = 1 THEN cp END) OVER (PARTITION BY a ORDER BY ts
                              RANGE BETWEEN coalesce(ts - first_in, INTERVAL 0 MINUTES) PRECEDING AND CURRENT ROW)) END AS g
            FROM s1
        ), s3 AS (                                       -- 거래마다: 자기를 근거로 삼는 트리거들의 g 최댓값
            SELECT edge_id, a AS subj,
                   CASE WHEN dir = 0 THEN                -- 입금: 뒤로 W 안의 모든 출금 트리거
                        max(g) OVER (PARTITION BY a ORDER BY ts RANGE BETWEEN CURRENT ROW AND {back} FOLLOWING)
                        WHEN last_in IS NOT NULL THEN    -- 출금: 직전 입금이 아직 기간 안에 있는 트리거까지만
                        max(g) OVER (PARTITION BY a ORDER BY ts RANGE BETWEEN CURRENT ROW
                              AND greatest(last_in + {back} - ts, INTERVAL 0 MINUTES) FOLLOWING) END AS v
            FROM s2)
        SELECT edge_id, max(v) AS v, arg_max(subj, (v, -subj)) AS subj       -- 세기가 같으면 번호가 작은 계좌
        FROM s3 WHERE v >= 1 GROUP BY edge_id""")
    return dict(표=table, 거래=_rows(con, table), 초=seconds)


# ----------------------------------------------------------------------------- ④ SCATTER-GATHER (2홉형)
def sg_candidates(con, tag):
    """SCATTER-GATHER 의 후보를 거래 시각과 무관하게 먼저 추린다. 필터마다 apply_filter() 뒤에 한 번 부른다.

    sg_cand_<tag> : 경유 계좌를 2개 이상 공유하는 (S, D) 와 그 경유 계좌 m.  S → m 거래와 m → D 거래가 둘 다 있다는 뜻이다
    sg_leg_<tag>  : 후보의 2단계 거래(m → D)마다, 그 시각까지의 마지막 1단계 거래(S → m) 시각 t1
    """
    started = time.perf_counter()
    con.execute(f'CREATE OR REPLACE TEMP TABLE sg_pairs AS SELECT DISTINCT src, dst FROM txn_{tag}')
    con.execute(f"""
        CREATE OR REPLACE TABLE sg_cand_{tag} AS
        SELECT p1.src AS s, p2.dst AS d, p1.dst AS m
        FROM sg_pairs p1 JOIN sg_pairs p2 ON p2.src = p1.dst
        WHERE p1.src <> p2.dst                                       -- S = D 는 순환이라 뺀다
        QUALIFY count(*) OVER (PARTITION BY p1.src, p2.dst) >= 2""")
    con.execute(f"""
        CREATE OR REPLACE TABLE sg_leg_{tag} AS
        WITH leg2 AS (SELECT c.s, c.d, c.m, e.edge_id AS e2, e.ts AS t2
                      FROM sg_cand_{tag} c JOIN txn_{tag} e ON e.src = c.m AND e.dst = c.d)
        SELECT l.s, l.d, l.m, l.e2, l.t2, e.ts AS t1
        FROM leg2 l ASOF JOIN txn_{tag} e ON e.src = l.s AND e.dst = l.m AND l.t2 >= e.ts""")
    pairs = con.execute(f'SELECT count(*) FROM (SELECT DISTINCT s, d FROM sg_cand_{tag})').fetchone()[0]
    return dict(tag=tag, 계좌_쌍=_rows(con, 'sg_pairs'), 후보_SD=pairs, 후보_2단계_거래=_rows(con, f'sg_leg_{tag}'),
                초=round(time.perf_counter() - started, 1))


def scatter_gather(con, W, tag):
    """SCATTER-GATHER: 출발 계좌 S → 경유 계좌 여러 개 → 도착 계좌 D.

    D 로 들어온 거래(경유 → D)가 트리거다. 트리거 시각 t0 에서 기간 W 만큼 돌아봐,
    "S → 경유" 가 "경유 → D" 보다 먼저(같은 분 포함) 있고 둘 다 기간 안인 경유 계좌 수 k 를 (S, D) 마다 센다.
    k >= M 이면 적중이고, 그 경유 계좌들의 S → 경유 거래와 경유 → D 거래가 근거 거래다.
    거래별 세기 v = 그 거래를 근거로 삼는 트리거들의 k 중 최댓값.
    """
    table = f'scatter_gather_{W}_{tag}'
    started = time.perf_counter()
    lower = f'INTERVAL {W - 1} MINUTES'
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sg_span AS         -- 트리거 × 경유 계좌 (그 경유 계좌의 마지막 2단계 시각 t2max)
        WITH trig AS (SELECT q.s, q.d, q.e2 AS trig, q.t2 AS t0, count(DISTINCT r.m) AS k
                      FROM sg_leg_{tag} q JOIN sg_leg_{tag} r
                        ON r.s = q.s AND r.d = q.d AND r.t2 <= q.t2 AND r.t1 >= q.t2 - {lower}
                      GROUP BY ALL HAVING count(DISTINCT r.m) >= 2)
        SELECT q.s, q.d, q.trig, q.t0, q.k, r.m, max(r.t2) AS t2max
        FROM trig q JOIN sg_leg_{tag} r ON r.s = q.s AND r.d = q.d AND r.t2 <= q.t0 AND r.t1 >= q.t0 - {lower}
        GROUP BY ALL""")
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH ev AS (
            SELECT r.e2 AS edge_id, p.k, p.d FROM sg_span p JOIN sg_leg_{tag} r          -- 2단계 근거: 경유 → D
              ON r.s = p.s AND r.d = p.d AND r.m = p.m AND r.t2 <= p.t0 AND r.t1 >= p.t0 - {lower}
            UNION ALL
            SELECT e.edge_id, p.k, p.d FROM sg_span p JOIN txn_{tag} e                   -- 1단계 근거: S → 경유
              ON e.src = p.s AND e.dst = p.m AND e.ts <= p.t2max AND e.ts >= p.t0 - {lower})
        SELECT edge_id, max(k) AS v, arg_max(d, (k, -d)) AS subj FROM ev GROUP BY edge_id""")
    return dict(표=table, 거래=_rows(con, table), 초=round(time.perf_counter() - started, 1))


# ----------------------------------------------------------------------------- ⑤⑥ CYCLE · RANDOM (경로형)
def links(con, W, tag, ratio_min, ratio_max):
    """이어지는 거래 쌍: 앞 거래의 수취 계좌가 뒤 거래의 송금 계좌이고, 뒤 거래가 앞 거래 이후(같은 분 포함) W 안이며,
    뒤 금액 ÷ 앞 금액(USD)이 [ratio_min, ratio_max] 안인 쌍. 경로는 이 쌍을 따라서만 이어진다."""
    table = f'links_{W}_{tag}'
    seconds = _timed(con, f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT e1.edge_id AS a, e2.edge_id AS b
        FROM txn_{tag} e1 JOIN txn_{tag} e2
          ON e2.src = e1.dst AND e2.ts >= e1.ts AND e2.ts <= e1.ts + INTERVAL {W - 1} MINUTES
         AND e2.usd BETWEEN e1.usd * {ratio_min} AND e1.usd * {ratio_max}""")
    return dict(표=table, 이어지는_쌍=_rows(con, table), 초=seconds)


def walk_counts(con, W, tag, max_hops):
    """경로를 실제로 만들기 전에 길이(거래 수)별 경로 수의 상한을 센다. 같은 계좌 재방문과 전체 기간 제한은 무시한 수다.
    합이 수천만을 넘으면 paths() 가 오래 걸리거나 메모리가 모자랄 수 있으니 금액 비 범위나 기간을 줄인다."""
    pairs = con.execute(f'SELECT a, b FROM links_{W}_{tag}').df()
    ids = np.unique(np.concatenate([pairs.a.to_numpy(), pairs.b.to_numpy()]))
    a, b = np.searchsorted(ids, pairs.a.to_numpy()), np.searchsorted(ids, pairs.b.to_numpy())
    count, out = np.ones(len(ids)), {}
    for hops in range(2, max_hops + 1):
        step = np.zeros(len(ids))
        np.add.at(step, a, count[b])
        count = step
        out[hops] = int(count.sum())
    return out


def paths(con, W, tag, max_hops):
    """links 를 따라 경로를 전부 만든 뒤 CYCLE 과 RANDOM 의 거래별 세기를 낸다.

    경로 규칙: 시간순(같은 분 포함), 첫 거래부터 마지막 거래까지 W 안, 같은 계좌를 두 번 지나지 않음,
               출발 계좌로 돌아오면 거기서 끝(순환), 거래 수는 max_hops 까지.
    CYCLE : 출발 계좌로 돌아온 경로의 거래.   v = 그 거래가 든 순환 중 가장 긴 것의 거래 수 (shortest 열에 가장 짧은 것)
    RANDOM: 돌아오지 않은 경로의 거래.       v = 그 거래가 든 경로 중 가장 긴 것의 거래 수
            순환의 앞부분도 닫히기 전까지는 돌아오지 않은 경로라서 RANDOM 에도 걸린다.
    """
    started = time.perf_counter()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE walks AS
        WITH RECURSIVE walk(origin, t_first, last_id, node, hops, accts, edges, closed) AS (
            SELECT e.src, e.ts, e.edge_id, e.dst, 1, [e.src, e.dst], [e.edge_id], false
            FROM txn_{tag} e WHERE e.edge_id IN (SELECT a FROM links_{W}_{tag})
            UNION ALL
            SELECT w.origin, w.t_first, n.edge_id, n.dst, w.hops + 1,
                   list_append(w.accts, n.dst), list_append(w.edges, n.edge_id), n.dst = w.origin
            FROM walk w JOIN links_{W}_{tag} l ON l.a = w.last_id JOIN txn_{tag} n ON n.edge_id = l.b
            WHERE NOT w.closed AND w.hops < {max_hops}
              AND n.ts <= w.t_first + INTERVAL {W - 1} MINUTES
              AND (n.dst = w.origin OR NOT list_contains(w.accts, n.dst)))
        SELECT origin, node, hops, edges, closed FROM walk WHERE hops >= 2""")
    n_cycle, n_open = con.execute(
        'SELECT count(*) FILTER (WHERE closed), count(*) FILTER (WHERE NOT closed) FROM walks').fetchone()
    con.execute(f"""
        CREATE OR REPLACE TABLE cycle_{W}_{tag} AS
        SELECT edge_id, max(hops) AS v, arg_max(origin, (hops, -origin)) AS subj, min(hops) AS shortest
        FROM (SELECT unnest(edges) AS edge_id, hops, origin FROM walks WHERE closed) GROUP BY edge_id""")
    con.execute(f"""
        CREATE OR REPLACE TABLE random_{W}_{tag} AS
        SELECT edge_id, max(hops) AS v, arg_max(node, (hops, -node)) AS subj
        FROM (SELECT unnest(edges) AS edge_id, hops, node FROM walks WHERE NOT closed) GROUP BY edge_id""")
    con.execute('DROP TABLE walks')
    return dict(순환=n_cycle, 돌아오지_않은_경로=n_open, 순환_거래=_rows(con, f'cycle_{W}_{tag}'),
                경로_거래=_rows(con, f'random_{W}_{tag}'), 초=round(time.perf_counter() - started, 1))


# ----------------------------------------------------------------------------- 조건을 바꿔 가며 재기
def sweep(con, grid, windows, filters, spans):
    """룰 × 필터 × 반복 × 기간 × 조건값 전부를 구간마다 잰다.

    grid: {룰: [조건값, ...]}, filters: {필터 이름: tag}, spans: {구간 이름: (시작일, 끝일)}
    지표에서 빼는 거래(skip)는 고른 거래에도 분모에도 넣지 않는다
    정밀도    = 고른 거래 중 세탁(패턴 종류 무관) 비율
    재현율    = 그 룰이 노리는 패턴의 세탁 거래 중 고른 비율
    F1        = 위 둘의 조화평균. 조건을 고를 때 쓴다
    대상_계좌 = 근거 거래가 걸린 서로 다른 대상 계좌 수
    """
    span = 'CASE ' + ' '.join(f"WHEN t.day {_between(w)} THEN '{name}'" for name, w in spans.items()) + ' END'
    own_total = {(s, kind): n for s, kind, n in con.execute(
        f'SELECT {span}, pattern_type, count(*) FROM tx_all t WHERE {span} IS NOT NULL AND NOT t.skip GROUP BY 1, 2').fetchall()}
    rows = []
    for rule, label in RULES.items():
        fan_rule = rule.startswith('fan_')
        once = 'r.v_once' if fan_rule else 'r.v'
        for name, tag in filters.items():
            for W in windows:
                source = f'{rule}_{W}_{tag} r JOIN tx_all t USING (edge_id) WHERE {span} IS NOT NULL AND NOT t.skip'
                # 세기 값마다의 건수를 한 번에 세고, 조건값별 합은 pandas 에서 낸다
                hist = con.execute(f"""SELECT {span} AS span, r.v AS v, {once} AS v_once, count(*) AS n, sum(t.y) AS laundering,
                                              count(*) FILTER (WHERE t.pattern_type = '{label}') AS own
                                       FROM {source} GROUP BY 1, 2, 3""").df()
                subj = con.execute(f"""SELECT span, v, v_once, count(*) AS n FROM (
                                           SELECT {span} AS span, r.subj, max(r.v) AS v, max({once}) AS v_once
                                           FROM {source} GROUP BY 1, 2) GROUP BY 1, 2, 3""").df()
                for s in spans:
                    h, a = hist[hist.span == s], subj[subj.span == s]
                    total = own_total.get((s, label), 0)
                    for repeat, column in (REPEAT if fan_rule else {'—': 'v'}).items():
                        for value in grid[rule]:
                            part = h[h[column] >= value]
                            picked, laundering, own = int(part.n.sum()), int(part.laundering.sum()), int(part.own.sum())
                            precision, recall = laundering / max(picked, 1), own / max(total, 1)
                            rows.append(dict(구간=s, 룰=label, 필터=name, 반복=repeat, 기간_분=W, 조건값=value,
                                             대상_계좌=int(a.n[a[column] >= value].sum()), 고른_거래=picked, 세탁=laundering,
                                             정밀도=round(precision, 4), 자기_패턴_거래=total, 자기_패턴_잡음=own, 재현율=round(recall, 4),
                                             F1=round(2 * precision * recall / max(precision + recall, 1e-12), 4)))
    return pd.DataFrame(rows)


def top(table, rule, n=8, by='F1'):
    """sweep 표에서 룰 하나의 상위 n 개 조건."""
    part = table[table.룰 == RULES[rule]].sort_values([by, '고른_거래'], ascending=[False, True]).head(n)
    return part.drop(columns=['구간', '룰', '자기_패턴_거래'], errors='ignore').reset_index(drop=True)


def view(table, rule, name, repeat=None, columns=('대상_계좌', '고른_거래', '정밀도', '재현율')):
    """sweep 표에서 룰 하나 · 필터 하나를 '조건값 × 기간' 모양으로 본다. FAN 은 repeat('허용' / '없음')도 고른다."""
    part = table[(table.룰 == RULES[rule]) & (table.필터 == name)]
    if repeat is not None:
        part = part[part.반복 == repeat]
    return pd.concat({W: g.set_index('조건값')[list(columns)] for W, g in part.groupby('기간_분')}, axis=1, names=['기간_분', None])


def pick(table, by='F1'):
    """룰마다 by 가 가장 큰 조건 한 벌을 고른다. 동점이면 고른 거래가 적은 쪽.
    돌려주는 것: {룰: dict(필터, 반복, 기간_분, 조건값)}"""
    best = table.sort_values([by, '고른_거래'], ascending=[False, True]).groupby('룰', sort=False).head(1).set_index('룰')
    return {rule: dict(필터=best.loc[label, '필터'], 반복=best.loc[label, '반복'],
                       기간_분=int(best.loc[label, '기간_분']), 조건값=int(best.loc[label, '조건값']))
            for rule, label in RULES.items()}


def chosen_rows(table, chosen):
    """sweep 표에서 고른 조건(chosen)의 줄만 룰 순서대로."""
    key = pd.DataFrame([dict(룰=RULES[rule], **condition) for rule, condition in chosen.items()])
    return key.merge(table, on=KEYS, how='left')


# ----------------------------------------------------------------------------- 고른 조건의 결과 모으기
def collect(con, chosen, filters, start, end):
    """[start, end] 날짜의 거래(tx_all)에, 고른 조건(chosen)에서 룰마다 걸렸는지(True / False)를 붙여 가져온다.
    열 이름은 패턴 이름('FAN-OUT', ...)이다. 지표에서 빼는 거래도 들어 있다(열 skip). 지표는 day_mask 로 만든 구간 안에서만 센다."""
    selects, joins = [], []
    for rule, c in chosen.items():
        column = REPEAT.get(c['반복'], 'v')
        selects.append(f"coalesce({rule}.{column} >= {c['조건값']}, false) AS \"{RULES[rule]}\"")
        joins.append(f"LEFT JOIN {rule}_{c['기간_분']}_{filters[c['필터']]} AS {rule} USING (edge_id)")
    return con.execute(f"""
        SELECT t.edge_id, t.day, t.y, t.pattern_type, t.attempt_id, t.skip, {', '.join(selects)}
        FROM tx_all t {' '.join(joins)}
        WHERE t.day BETWEEN DATE '{start}' AND DATE '{end}'
        ORDER BY t.ts, t.edge_id""").df()


def day_mask(frame, window):
    """window 날짜 안이고 지표에서 빼지 않는 거래. 지표 함수들은 이 구간 안에서만 센다."""
    day = pd.to_datetime(frame.day)
    return ((day >= pd.Timestamp(window[0])) & (day <= pd.Timestamp(window[1])) & ~frame.skip.astype(bool)).to_numpy()


def method_metrics(name, chosen, frame, window):
    """고른 거래 집합(chosen: bool 배열) 하나의 결과. window 는 day_mask 로 만든 구간이다.

    재현율 = 구간 안 세탁 거래 중 고른 비율. 지표에서 빼는 거래는 구간에 없다.
    구간에 무패턴 세탁이나 룰이 없는 패턴이 남아 있으면 분모를 좁힌 재현율 열을 더 낸다.
    """
    y, kind = frame.y.to_numpy(), frame.pattern_type.to_numpy()
    chosen = chosen & window
    positive = window & (y == 1)
    patterned = positive & (kind != 'NO_PATTERN')
    targeted = positive & np.isin(kind, list(RULES.values()))
    tp = int((chosen & positive).sum())
    row = dict(방법=name, 고른_거래=int(chosen.sum()), 세탁=tp, 정상=int(chosen.sum()) - tp,
               정밀도=round(tp / max(int(chosen.sum()), 1), 4),
               재현율=round(tp / max(int(positive.sum()), 1), 4))
    if patterned.sum() != positive.sum():
        row['재현율_패턴_전체'] = round(int((chosen & patterned).sum()) / max(int(patterned.sum()), 1), 4)
    if targeted.sum() != positive.sum():
        row['재현율_룰이_있는_6종'] = round(int((chosen & targeted).sum()) / max(int(targeted.sum()), 1), 4)
    return row


def by_pattern(frame, window, methods):
    """패턴별 재현율. methods: {이름: bool 배열}"""
    y, kind = frame.y.to_numpy(), frame.pattern_type.to_numpy()
    rows = []
    for name in sorted(set(kind[window & (y == 1)])):
        base = window & (y == 1) & (kind == name)
        row = dict(패턴=name, 세탁_거래=int(base.sum()))
        row.update({label: round(int((base & mask).sum()) / int(base.sum()), 4) for label, mask in methods.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def by_attempt(frame, window, methods, exclude=()):
    """세탁 시도 단위. 시도마다 window 안 거래 중 고른 수를 세고, 패턴별로 1건 이상 / 절반 이상 / 전부 잡은 시도의 비율을 낸다."""
    keep = np.flatnonzero(window & frame.attempt_id.notna().to_numpy() & ~frame.pattern_type.isin(list(exclude)).to_numpy())
    inside = frame.iloc[keep][['pattern_type', 'attempt_id']]
    rows = []
    for label, mask in methods.items():
        per = inside.assign(hit=mask[keep]).groupby(['pattern_type', 'attempt_id']).agg(size=('hit', 'size'), hit=('hit', 'sum')).reset_index()
        for kind, g in per.groupby('pattern_type'):
            rows.append(dict(패턴=kind, 방법=label, 시도=len(g), 한_건_이상=round(float((g.hit >= 1).mean()), 3),
                             절반_이상=round(float((g.hit * 2 >= g['size']).mean()), 3), 전부=round(float((g.hit == g['size']).mean()), 3)))
    return pd.DataFrame(rows).sort_values(['패턴', '방법'], kind='stable').reset_index(drop=True)
