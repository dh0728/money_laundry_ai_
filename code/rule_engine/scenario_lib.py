"""일반 템플릿(Oracle 시나리오 대응) 4종 실험 도구 (07_scenario_rules).

패턴 룰(rule_lib)과 같은 원장 · 같은 규칙으로 일반 룰 4종을 돌리고, 조건 후보마다 고른 거래를 정답 라벨로 잰다.
rule_lib 이 만든 표를 그대로 쓴다.
  tx_all      : 읽은 거래 전부 (지표는 여기서 센다)
  tx_<tag>    : 거래 필터를 통과한 거래 (rule_lib.apply_filter). 자기 자신에게 보낸 거래는 없다
  busy        : 바쁜 계좌 (rule_lib.busy_accounts)

룰 4종 (Oracle 시나리오 번호는 '룰-템플릿-정의' 문서 5.1 의 번호)
  단건 고액거래   (#3)  거래 1건의 금액 >= 기준
  분할 입 · 출금  (#20) 계좌가 기간 안에 "보고 기준 바로 아래 금액"의 거래를 N 건 이상
  같은 금액 반복  (#9)  계좌가 기간 안에 검사 거래와 같은(±오차) 금액의 거래를 N 건 이상
  급속 자금 통과  (#1)  계좌가 기간 안에 받은 합계가 기준 이상이고, 첫 입금 뒤 내보낸 합계가 받은 합계의 일정 비율 이상

공통 규칙 (rule_lib 과 같다)
  - 기간은 트리거 시각을 포함한 직전 W 분: [t0 - (W-1)분, t0]. 같은 분의 거래는 기간 안이다
  - 바쁜 계좌는 대상이 되지 않는다 (단건 고액거래는 계좌를 돌아보지 않으므로 해당 없음)
  - 금액은 보낸 금액을 고정 환율로 바꾼 USD
  - 근거 거래 = 적중한 트리거의 기간 안에서 그 조건을 이루는 거래. 이것이 룰이 "고른 거래"다

후보(candidate) 하나 = (표, 그 표에서 고른 거래를 가리는 식). 룰 함수가 표를 만들고 *_candidates 가 후보 목록을 만든다.
measure() 가 후보 전부를 구간마다 잰다. 지표의 뜻은 measure() 설명에 있다.

검증: 가짜 원장에서 네 룰의 SQL 결과가 같은 정의를 Python 반복문으로 짠 계산과 일치하는 것을 확인했다(2026-11-24). 실제 데이터로는 돌려 보지 않았다.
"""
import time

import pandas as pd

LABELS = {'large_tx': '단건 고액거래', 'structuring': '분할 입·출금', 'same_amount': '같은 금액 반복', 'pass_through': '급속 자금 통과'}
SIDES = {'in': '입금', 'out': '출금'}
INFO = ['룰', '방향', '필터', '기간_분', '조건']          # 후보 하나를 사람이 읽는 열


def _timed(con, sql):
    started = time.perf_counter()
    con.execute(sql)
    return round(time.perf_counter() - started, 1)


def _rows(con, table):
    return con.execute(f'SELECT count(*) FROM {table}').fetchone()[0]


def _code(value):
    """표 이름 · 열 이름에 넣을 수 있게 숫자를 글자로. 0.05 → 0p05, 10000 → 10000"""
    return f'{value:g}'.replace('.', 'p').replace('-', 'm').replace('+', '')


def _side(side):
    """방향 → (대상 계좌 열, 상대 계좌 열). 'in' = 받는 쪽이 대상, 'out' = 보내는 쪽이 대상"""
    return ('dst', 'src') if side == 'in' else ('src', 'dst')


def _between(window):
    return f"BETWEEN DATE '{window[0]}' AND DATE '{window[1]}'"


# ----------------------------------------------------------------------------- ① 단건 고액거래
def large_tx(con, tag, floor_usd):
    """단건 고액거래: 거래 1건의 금액이 기준 이상이면 그 거래를 고른다. 돌아보는 기간도 대상 계좌도 없다.

    표 large_tx_<tag>: 금액이 floor_usd(후보 중 가장 작은 기준) 이상인 거래. 열 edge_id, subj(송금 계좌), usd
    후보의 식은 usd >= 기준 이다.
    """
    table = f'large_tx_{tag}'
    seconds = _timed(con, f'CREATE OR REPLACE TABLE {table} AS SELECT edge_id, src AS subj, usd FROM tx_{tag} WHERE usd >= {floor_usd}')
    return dict(표=table, 거래=_rows(con, table), 초=seconds)


def large_candidates(filters, tags, thresholds):
    return [dict(룰=LABELS['large_tx'], 방향='—', 필터=name, 기간_분=0, 조건=f'건당 {value:,} USD 이상',
                 표=f'large_tx_{tags[name]}', 식=f'r.usd >= {value}', 유일=True, 건당_최소=value)
            for name in filters for value in thresholds]


# ----------------------------------------------------------------------------- ② 분할 입 · 출금
def structuring(con, side, W, low, high, tag):
    """분할 입 · 출금: 대상 계좌가 기간 안에 금액이 [low, high) 인 거래를 N 건 이상 했다.

    high 는 보고 기준(그 이상이면 보고 대상인 금액), low 는 "기준 바로 아래"로 볼 하한이다.
    대상 계좌의 띠 안 거래 하나하나가 트리거다. 트리거 시각에서 기간 W 만큼 돌아봐 띠 안 거래 수 n 을 센다(자기 포함).
    n >= N 인 트리거의 기간 안에 든 띠 안 거래 전부가 근거 거래다.
      v = 그 거래를 기간 안에 담는 트리거들의 n 중 최댓값.  따라서 v >= N 인 거래가 조건값 N 의 근거 거래다.
    띠 밖의 거래는 세지도 않고 근거 거래도 되지 않는다.
    """
    subject, _ = _side(side)
    table = f'structuring_{side}_{W}_{_code(low)}_{tag}'
    back = f'INTERVAL {W - 1} MINUTES'
    seconds = _timed(con, f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH trig AS (                                   -- 트리거마다: 기간 안 띠 안 거래 수
            SELECT edge_id, {subject} AS a, ts, count(*) OVER w AS n
            FROM tx_{tag} WHERE usd >= {low} AND usd < {high} AND {subject} NOT IN (SELECT a FROM busy)
            WINDOW w AS (PARTITION BY {subject} ORDER BY ts RANGE BETWEEN {back} PRECEDING AND CURRENT ROW)
        ), reach AS (                                    -- 거래마다: 자기를 담는 트리거들의 n 최댓값
            SELECT edge_id, a AS subj, max(n) OVER (PARTITION BY a ORDER BY ts RANGE BETWEEN CURRENT ROW AND {back} FOLLOWING) AS v
            FROM trig)
        SELECT edge_id, subj, v FROM reach WHERE v >= 2""")
    return dict(표=table, 거래=_rows(con, table), 초=seconds)


def structuring_candidates(sides, filters, tags, windows, lows, high, counts):
    return [dict(룰=LABELS['structuring'], 방향=SIDES[side], 필터=name, 기간_분=W,
                 조건=f'건당 {low:,} 이상 {high:,} USD 미만, {count}건 이상',
                 표=f'structuring_{side}_{W}_{_code(low)}_{tags[name]}', 식=f'r.v >= {count}', 유일=True,
                 건당_하한=low, 건당_상한=high, 건수=count)
            for side in sides for name in filters for W in windows for low in lows for count in counts]


# ----------------------------------------------------------------------------- ③ 같은 금액 반복
def same_amount(con, side, windows, tolerances, tag):
    """같은 금액 반복: 대상 계좌가 기간 안에 검사 거래와 같은(±오차) 금액의 거래를 N 건 이상 했다.

    대상 계좌의 거래 하나하나가 트리거다. 트리거 금액을 x, 오차를 tol 이라 하면
      n = 기간 W 안의 같은 방향 거래 중 금액이 [x * (1 - tol), x * (1 + tol)] 인 것의 수 (트리거 자신 포함)
    n >= N 인 트리거에서, 그 금액 범위에 든 기간 안 거래 전부가 근거 거래다.
      v_<W>_<tol> = 그 거래를 근거로 삼는 트리거들의 n 중 최댓값
    tol = 0 이면 USD 환산 금액이 정확히 같은 거래만 센다.

    만드는 방법: 같은 계좌의 거래 쌍을 (가장 긴 기간, 가장 큰 오차)로 한 번 잇고, 기간 · 오차 조합별 수는 그 안에서 조건을 걸어 같이 센다.
      1) 후보 추리기: 같은 계좌에 금액이 가장 가까운 이웃이 최대 오차 밖인 거래는 어떤 쌍에도 못 들어가므로 미리 뺀다
      2) 트리거마다 n 세기   3) 거래마다 v 세기 (n >= 2 인 트리거만)
    쌍을 표로 만들지 않고 바로 집계하므로 메모리는 트리거 수만큼만 쓴다. 대신 같은 조인을 두 번 한다.
    표 same_amount_<side>_<tag>: edge_id, subj, v_<W>_<tol 코드> ...
    """
    subject, _ = _side(side)
    table = f'same_amount_{side}_{tag}'
    W_max, T_max = max(windows), max(tolerances)
    started = time.perf_counter()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sa_base AS
        WITH b AS (SELECT edge_id, {subject} AS a, ts, usd FROM tx_{tag}
                   WHERE usd > 0 AND {subject} NOT IN (SELECT a FROM busy)),
        n AS (SELECT *, lag(usd) OVER o AS below, lead(usd) OVER o AS above FROM b WINDOW o AS (PARTITION BY a ORDER BY usd))
        SELECT edge_id, a, ts, usd FROM n          -- 금액이 가장 가까운 이웃이 최대 오차 안에 있는 거래만
        WHERE below >= usd * (1 - {T_max}) OR above * (1 - {T_max}) <= usd""")
    pair = f"""e.a = t.a AND e.ts <= t.ts AND e.ts >= t.ts - INTERVAL {W_max - 1} MINUTES
               AND e.usd BETWEEN t.usd * (1 - {T_max}) AND t.usd * (1 + {T_max})"""
    combos = [(W, tol, f'{W}_{_code(tol)}',
               f'e.ts >= t.ts - INTERVAL {W - 1} MINUTES AND e.usd BETWEEN t.usd * (1 - {tol}) AND t.usd * (1 + {tol})')
              for W in windows for tol in tolerances]
    counts = ', '.join(f'count(*) FILTER (WHERE {inside}) AS n_{code}' for _, _, code, inside in combos)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sa_trig AS         -- 트리거마다: 기간 · 오차 조합별 같은 금액 거래 수 (자기 포함)
        SELECT t.edge_id AS trig, {counts}
        FROM sa_base t JOIN sa_base e ON {pair}
        GROUP BY t.edge_id HAVING count(*) >= 2""")
    reach = ', '.join(f'coalesce(max(g.n_{code}) FILTER (WHERE {inside}), 0) AS v_{code}' for _, _, code, inside in combos)
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS              -- 거래마다: 자기를 근거로 삼는 트리거들의 n 최댓값
        SELECT e.edge_id, any_value(e.a) AS subj, {reach}
        FROM sa_trig g JOIN sa_base t ON t.edge_id = g.trig JOIN sa_base e ON {pair}
        GROUP BY e.edge_id""")
    out = dict(표=table, 후보_거래=_rows(con, 'sa_base'), 트리거=_rows(con, 'sa_trig'), 거래=_rows(con, table))
    con.execute('DROP TABLE sa_base; DROP TABLE sa_trig')
    return dict(**out, 초=round(time.perf_counter() - started, 1))


def same_candidates(sides, filters, tags, windows, tolerances, counts):
    return [dict(룰=LABELS['same_amount'], 방향=SIDES[side], 필터=name, 기간_분=W,
                 조건=('같은 금액' if tol == 0 else f'±{tol:.0%} 금액') + f', {count}건 이상',
                 표=f'same_amount_{side}_{tags[name]}', 식=f'r.v_{W}_{_code(tol)} >= {count}', 유일=True, 오차=tol, 건수=count)
            for side in sides for name in filters for W in windows for tol in tolerances for count in counts]


# ----------------------------------------------------------------------------- ④ 급속 자금 통과
def pass_combos(sums, ratios, caps):
    """급속 자금 통과의 조건 조합. caps 의 True 는 "비율 상한을 둔다"(상한 = 1 ÷ 하한), False 는 상한 없음."""
    return [dict(최소_입금합계=s, 비율_하한=r, 비율_상한=(round(1 / r, 4) if cap else None)) for s in sums for r in ratios for cap in caps]


def _pass_hit(combo):
    cap = '' if combo['비율_상한'] is None else f" AND sum_out <= sum_in * {combo['비율_상한']}"
    return f"dir = 1 AND sum_in >= {combo['최소_입금합계']} AND sum_out >= sum_in * {combo['비율_하한']}{cap}"


def pass_through(con, W, tag, combos):
    """급속 자금 통과: 대상 계좌가 받은 돈을 기간 안에 (거의) 그대로 내보냈다.

    대상 계좌 A 의 출금 하나하나가 트리거다. 트리거 시각 t0 에서 기간 W 만큼 돌아봐
      sum_in  = 기간 안 입금 합계
      sum_out = 기간 안 첫 입금 시각부터 t0 까지 출금 합계   ("받은 뒤에 내보낸다". GATHER-SCATTER 와 같은 순서 규칙)
    를 낸다. 조합(combo)마다  sum_in >= 최소_입금합계  AND  sum_out >= sum_in * 비율_하한  [AND sum_out <= sum_in * 비율_상한]  이면 적중이다.
    근거 거래 = 기간 안 입금 전부 + 첫 입금 뒤의 출금.
    상대 계좌 수는 보지 않는다(GATHER-SCATTER 와 다른 점). 한 계좌에서 받아 한 계좌로 보내도 걸린다.

    표 pass_through_<W>_<tag>: edge_id, subj, c0, c1, ... (c{i} = combos[i] 에서 근거 거래인가)
    한 거래가 보내는 계좌와 받는 계좌 양쪽에서 근거 거래가 될 수 있어 edge_id 가 두 줄일 수 있다.
    """
    table = f'pass_through_{W}_{tag}'
    back = f'INTERVAL {W - 1} MINUTES'
    started = time.perf_counter()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pt_ev AS
        WITH ev AS (                                     -- 계좌 쪽에서 본 거래. dir 0 = 받음, 1 = 보냄
            SELECT edge_id, dst AS a, ts, 0 AS dir, usd FROM tx_{tag} WHERE dst NOT IN (SELECT a FROM busy)
            UNION ALL
            SELECT edge_id, src, ts, 1, usd FROM tx_{tag} WHERE src NOT IN (SELECT a FROM busy)
        ), s1 AS (
            SELECT *,
                   sum(CASE WHEN dir = 0 THEN usd END) OVER w AS sum_in,
                   min(CASE WHEN dir = 0 THEN ts END) OVER w AS first_in,
                   max(CASE WHEN dir = 0 THEN ts END) OVER (PARTITION BY a ORDER BY ts
                         RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_in      -- 이 시각까지의 마지막 입금
            FROM ev
            WINDOW w AS (PARTITION BY a ORDER BY ts RANGE BETWEEN {back} PRECEDING AND CURRENT ROW))
        SELECT edge_id, a, ts, dir, last_in, sum_in,
               CASE WHEN dir = 1 AND first_in IS NOT NULL THEN
                    sum(CASE WHEN dir = 1 THEN usd END) OVER (PARTITION BY a ORDER BY ts
                          RANGE BETWEEN coalesce(ts - first_in, INTERVAL 0 MINUTES) PRECEDING AND CURRENT ROW) END AS sum_out
        FROM s1""")
    # 가장 느슨한 조합에도 안 걸리는 계좌는 어느 조합에서도 안 걸린다. 그런 계좌를 먼저 뺀다
    loosest = dict(최소_입금합계=min(c['최소_입금합계'] for c in combos), 비율_하한=min(c['비율_하한'] for c in combos), 비율_상한=None)
    con.execute(f'CREATE OR REPLACE TEMP TABLE pt_keep AS SELECT DISTINCT a FROM pt_ev WHERE {_pass_hit(loosest)}')
    # 거래마다: 자기를 근거로 삼는 적중 트리거가 있는가. 입금은 뒤로 W 안의 트리거, 출금은 직전 입금이 아직 기간 안에 있는 트리거까지
    reach = f"""RANGE BETWEEN CURRENT ROW AND CASE WHEN dir = 0 THEN {back}
                     ELSE greatest(coalesce(last_in + {back} - ts, INTERVAL 0 MINUTES), INTERVAL 0 MINUTES) END FOLLOWING"""
    flags = ', '.join(f'coalesce(max(CASE WHEN {_pass_hit(c)} THEN 1 ELSE 0 END) OVER fwd, 0) = 1 AS c{i}' for i, c in enumerate(combos))
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        WITH f AS (SELECT edge_id, a AS subj, {flags}
                   FROM pt_ev WHERE a IN (SELECT a FROM pt_keep)
                   WINDOW fwd AS (PARTITION BY a ORDER BY ts {reach}))
        SELECT * FROM f WHERE {' OR '.join(f'c{i}' for i in range(len(combos)))}""")
    out = dict(표=table, 계좌_쪽_거래=_rows(con, 'pt_ev'), 남긴_계좌=_rows(con, 'pt_keep'), 줄=_rows(con, table))
    con.execute('DROP TABLE pt_ev; DROP TABLE pt_keep')
    return dict(**out, 초=round(time.perf_counter() - started, 1))


def pass_candidates(filters, tags, windows, combos):
    def text(c):
        cap = '' if c['비율_상한'] is None else f" ~ {c['비율_상한']:.0%}"
        return f"받은 합계 {c['최소_입금합계']:,} USD 이상, 출금 ÷ 입금 {c['비율_하한']:.0%}{cap or ' 이상'}"
    return [dict(룰=LABELS['pass_through'], 방향='—', 필터=name, 기간_분=W, 조건=text(c),
                 표=f'pass_through_{W}_{tags[name]}', 식=f'r.c{i}', 유일=False, **c)
            for name in filters for W in windows for i, c in enumerate(combos)]


# ----------------------------------------------------------------------------- 후보를 구간마다 재기
def measure(con, candidates, spans, chunk=40):
    """후보 전부를 구간마다 잰다. 돌려주는 표의 한 줄 = (후보, 구간).

    candidates: *_candidates 가 만든 목록 (순서가 후보 번호다), spans: {구간 이름: (시작일, 끝일)}
    지표에서 빼는 거래(tx_all.skip)는 고른 거래에도 분모에도 넣지 않는다 (rule_lib.sweep 과 같다)
      대상_계좌 = 근거 거래가 걸린 서로 다른 대상 계좌 수 (단건 고액거래는 송금 계좌 수)
      고른_거래 = 룰이 고른 거래(근거 거래) 수
      세탁      = 그중 세탁 거래 수 (패턴 종류 무관)
      정밀도    = 세탁 ÷ 고른_거래
      재현율    = 세탁 ÷ 구간 안 세탁 거래 전체.  일반 룰은 노리는 패턴이 따로 없어서 분모가 6종 패턴 세탁 전체다
      F1        = 위 둘의 조화평균
      하루_평균 = 고른_거래 ÷ 구간 날짜 수.  원장_대비 = 고른_거래 ÷ 구간 안 거래 전체
      뺀_세탁   = 고른 거래 중 지표에서 뺀 세탁(무패턴 등)의 수. 지표에는 안 들어가고 참고로만 본다
    """
    span = 'CASE ' + ' '.join(f"WHEN day {_between(w)} THEN '{name}'" for name, w in spans.items()) + ' END'
    con.execute(f"""CREATE OR REPLACE TEMP TABLE tx_span AS
                    SELECT edge_id, {span} AS span, y, skip FROM tx_all WHERE {span} IS NOT NULL""")
    total = {s: (n, int(l)) for s, n, l in con.execute(
        'SELECT span, count(*) FILTER (WHERE NOT skip), coalesce(sum(y) FILTER (WHERE NOT skip), 0) FROM tx_span GROUP BY 1').fetchall()}
    days = {name: (pd.Timestamp(w[1]) - pd.Timestamp(w[0])).days + 1 for name, w in spans.items()}
    by_table = {}
    for number, c in enumerate(candidates):
        by_table.setdefault(c['표'], []).append(number)
    rows = []
    for table, numbers in by_table.items():
        for start in range(0, len(numbers), chunk):
            part = numbers[start:start + chunk]
            unique = candidates[part[0]]['유일']
            subject = ', '.join(f"count(DISTINCT r.subj) FILTER (WHERE ({candidates[n]['식']}) AND NOT t.skip) AS s{i}" for i, n in enumerate(part))
            if unique:           # edge_id 가 한 줄뿐인 표: 바로 센다
                cols = ', '.join(f"""count(*) FILTER (WHERE ({candidates[n]['식']}) AND NOT t.skip) AS p{i},
                                     coalesce(sum(t.y) FILTER (WHERE ({candidates[n]['식']}) AND NOT t.skip), 0) AS l{i},
                                     count(*) FILTER (WHERE ({candidates[n]['식']}) AND t.skip) AS x{i}""" for i, n in enumerate(part))
                got = con.execute(f'SELECT t.span AS span, {cols}, {subject} FROM {table} r JOIN tx_span t USING (edge_id) GROUP BY 1').df()
            else:                # 같은 거래가 여러 줄일 수 있는 표: 거래마다 먼저 합친 뒤 센다
                flags = ', '.join(f"bool_or({candidates[n]['식']}) AS f{i}" for i, n in enumerate(part))
                cols = ', '.join(f"""count(*) FILTER (WHERE f{i} AND NOT skip) AS p{i},
                                     coalesce(sum(y) FILTER (WHERE f{i} AND NOT skip), 0) AS l{i},
                                     count(*) FILTER (WHERE f{i} AND skip) AS x{i}""" for i in range(len(part)))
                got = con.execute(f"""SELECT span, {cols} FROM (
                                          SELECT r.edge_id, any_value(t.span) AS span, any_value(t.y) AS y, any_value(t.skip) AS skip, {flags}
                                          FROM {table} r JOIN tx_span t USING (edge_id) GROUP BY r.edge_id) GROUP BY 1""").df()
                got = got.merge(con.execute(f'SELECT t.span AS span, {subject} FROM {table} r JOIN tx_span t USING (edge_id) GROUP BY 1').df(), on='span', how='left')
            got = got.set_index('span')
            for s in spans:
                n_all, n_laundering = total.get(s, (0, 0))
                for i, n in enumerate(part):
                    value = (lambda name: int(got.at[s, f'{name}{i}']) if s in got.index else 0)
                    picked, laundering = value('p'), value('l')
                    precision, recall = laundering / max(picked, 1), laundering / max(n_laundering, 1)
                    c = candidates[n]
                    rows.append(dict(후보=n, 구간=s, **{k: c[k] for k in INFO}, 대상_계좌=value('s'), 고른_거래=picked, 세탁=laundering,
                                     정밀도=round(precision, 4), 재현율=round(recall, 4),
                                     F1=round(2 * precision * recall / max(precision + recall, 1e-12), 4),
                                     하루_평균=round(picked / days[s], 1), 원장_대비=round(picked / max(n_all, 1), 6), 뺀_세탁=value('x')))
    return pd.DataFrame(rows).sort_values(['구간', '후보']).reset_index(drop=True)


def top(table, rule, n=10, by='F1', min_picked=0):
    """measure 표(한 구간)에서 룰 하나의 상위 n 개 후보. min_picked 로 고른 거래가 너무 적은 후보를 뺄 수 있다."""
    part = table[(table.룰 == LABELS[rule]) & (table.고른_거래 >= min_picked)]
    return part.sort_values([by, '고른_거래'], ascending=[False, True]).head(n).drop(columns=['구간', '룰'], errors='ignore').reset_index(drop=True)


def pick(table, by='F1', min_picked=1):
    """룰마다 by 가 가장 큰 후보의 번호를 고른다. 동점이면 고른 거래가 적은 쪽. 고른 거래가 min_picked 보다 적은 후보는 뺀다.
    돌려주는 것: {룰 이름(영문): 후보 번호}"""
    part = table[table.고른_거래 >= min_picked].sort_values([by, '고른_거래'], ascending=[False, True])
    best = part.groupby('룰', sort=False).head(1).set_index('룰').후보
    return {rule: int(best[label]) for rule, label in LABELS.items() if label in best.index}


def find(candidates, rule, **conditions):
    """조건으로 후보 번호를 찾는다. 예: find(CANDS, 'structuring', 방향='입금', 필터='없음', 기간_분=1440, 건당_하한=8000, 건수=3)
    딱 하나만 맞아야 한다."""
    found = [n for n, c in enumerate(candidates)
             if c['룰'] == LABELS[rule] and all(c.get(key) == value for key, value in conditions.items())]
    assert len(found) == 1, f'{LABELS[rule]} {conditions}: 맞는 후보가 {len(found)}개다 (1개여야 한다)'
    return found[0]


def chosen_rows(table, chosen):
    """measure 표에서 고른 후보(chosen: {룰: 후보 번호})의 줄만 룰 순서대로."""
    order = [chosen[rule] for rule in LABELS if rule in chosen]
    return table.set_index('후보').loc[order].reset_index()


# ----------------------------------------------------------------------------- 고른 후보의 결과 모으기
def collect(con, candidates, chosen, start, end):
    """[start, end] 날짜의 거래(tx_all)에, 고른 후보(chosen: {룰: 후보 번호})에서 룰마다 걸렸는지(True / False)를 붙여 가져온다.
    열 이름은 룰의 한글 이름이다. 지표에서 빼는 거래도 들어 있다(열 skip). 줄 순서는 시각, edge_id 순이다 (rule_lib.collect 와 같다)."""
    selects, joins = [], []
    for i, (rule, number) in enumerate(chosen.items()):
        c = candidates[number]
        selects.append(f'coalesce(q{i}.hit, false) AS "{LABELS[rule]}"')
        joins.append(f"LEFT JOIN (SELECT r.edge_id, true AS hit FROM {c['표']} r WHERE {c['식']} GROUP BY r.edge_id) AS q{i} USING (edge_id)")
    return con.execute(f"""
        SELECT {', '.join(['t.edge_id', 't.day', 't.y', 't.pattern_type', 't.attempt_id', 't.skip', *selects])}
        FROM tx_all t {' '.join(joins)}
        WHERE t.day BETWEEN DATE '{start}' AND DATE '{end}'
        ORDER BY t.ts, t.edge_id""").df()
