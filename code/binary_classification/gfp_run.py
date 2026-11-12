"""GFP(snapml GraphFeaturePreprocessor) 모티프 피처를 커널 밖에서 계산한다 — 05 노트북 3절의 스크립트 판.

왜 스크립트인가: Jupyter 커널 안에서 돌리면 snapml 이 크래시할 때 커널째 죽어 메시지를 못 본다 (RSS 3.3GB 에서 비결정적으로 사망).
여기서는 faulthandler 로 세그폴트 위치를 로그에 남기고, part 단위로 재개한다.

사용:
  nohup python gfp_run.py run --threads 8 > results/HI-Large/gfp/gfp_run.log 2>&1 &
  python gfp_run.py bench --start 4000000 --rows 600000 --threads 8 --cycle-tw 21600 --lc-len 4   # 빽빽한 구간에서 정상 속도 측정
옵션:
  --threads N        snapml num_threads (크래시 재현용으로 1 도 시도)
  --cycle-tw SEC     temp-cycle / lc-cycle 시간 창 (기본 86400). 6시간 = 21600
  --lc-len K         lc-cycle 최대 길이 (기본 6)
  --tag NAME         출력 폴더 접미사 (설정을 바꿔 돌릴 때 part 가 섞이지 않게). 기본 ''
  --tw SEC           전체 time_window 와 fan/degree 창 (기본 86400)
  --no-vs            vertex stats 끄기 (속도 비교용)
자기 루프(src == dst) 는 GFP 입력에서 뺀다 — 창 밖 삭제 때 'Edge does not exist' 로 크래시하는 원인이고 모티프에도 무의미. 그 행의 피처는 0
"""
import argparse, faulthandler, gc, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil

faulthandler.enable(all_threads=True)
sys.path.append(str(Path(__file__).resolve().parent))
import bc_lib as B

DAY = 86400
READ_CHUNK = 2_000_000
WARMUP_ROWS = 1_500_000
PATTERN_KEYS = ['fan', 'degree', 'scatter-gather', 'temp-cycle', 'lc-cycle', 'vertex_stats']
FAN_ORDER = ('in', 'out')            # 05 노트북 1절에서 확인한 순서
VS_FEATS = ['fan', 'deg', 'ratio', 'avg', 'sum', 'min', 'max']


def gfp_params(threads, cycle_tw, lc_len, tw=DAY, vs=True):
    return {
        'num_threads': threads, 'time_window': tw,
        'vertex_stats': vs, 'vertex_stats_cols': [4], 'vertex_stats_feats': [0, 1, 2, 3, 4, 5, 6],
        'fan': True, 'fan_tw': tw, 'fan_bins': [2, 3, 4, 5, 6, 8, 10, 15],
        'degree': True, 'degree_tw': tw, 'degree_bins': [2, 3, 4, 5, 6, 8, 10, 15],
        'scatter-gather': True, 'scatter-gather_tw': 6 * 3600, 'scatter-gather_bins': [2, 3, 4, 6, 8],
        'temp-cycle': True, 'temp-cycle_tw': cycle_tw, 'temp-cycle_bins': [2, 3, 4, 5, 6],
        'lc-cycle': True, 'lc-cycle_tw': cycle_tw, 'lc-cycle_len': lc_len, 'lc-cycle_bins': [2, 3, 4, 5, 6],
    }


def gfp_names(p):
    """05 노트북 1절과 같은 규칙. bin 수 = len(bins) (각 bin 이 열 하나), fan/degree 는 in → out, vertex stats 는 피처 × 4 조합"""
    names = []
    for k in PATTERN_KEYS:
        if k in ('fan', 'degree'):
            tag = 'fan' if k == 'fan' else 'deg'; nb = len(p[f'{k}_bins'])
            names += [f'gfp_{tag}_{FAN_ORDER[0]}_b{i}' for i in range(nb)] + [f'gfp_{tag}_{FAN_ORDER[1]}_b{i}' for i in range(nb)]
        elif k == 'vertex_stats':
            if not p['vertex_stats']:
                continue
            for f in p['vertex_stats_feats']:
                names += [f'gfp_vs_{VS_FEATS[f]}_{j}' for j in range(4)]
        else:
            tag = {'scatter-gather': 'sg', 'temp-cycle': 'tcyc', 'lc-cycle': 'lccyc'}[k]
            names += [f'gfp_{tag}_b{i}' for i in range(len(p[f'{k}_bins']))]
    return names


def iter_chunks(pf, cols, start_row=0, n_rows=None, chunk=READ_CHUNK):
    seen, taken = 0, 0
    for batch in pf.iter_batches(batch_size=chunk, columns=cols):
        n = batch.num_rows
        if seen + n <= start_row:
            seen += n; continue
        arr = np.column_stack([batch.column(c).to_numpy().astype(np.float64) for c in cols])
        if seen < start_row:
            arr = arr[start_row - seen:]
        seen += n
        if n_rows is not None and taken + len(arr) > n_rows:
            arr = arr[:n_rows - taken]
        taken += len(arr); yield arr
        if n_rows is not None and taken >= n_rows:
            return


def transform_by_minute(gp, X, fitted, n_feat, log=None, log_every=50):
    """X 를 같은 timestamp 묶음으로 잘라 순서대로 transform. 자기 루프(src == dst) 는 입력에서 빼고 피처 0 으로 둔다"""
    out = np.zeros((len(X), n_feat), np.float64); t0 = time.time()
    keep = np.flatnonzero(X[:, 1] != X[:, 2]); Xk = X[keep]
    ts = Xk[:, 3]; assert (np.diff(ts) >= 0).all(), '시간순 아님'
    bounds = np.flatnonzero(np.diff(ts) != 0) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, len(Xk)]
    for k, (s, e) in enumerate(zip(starts, ends)):
        res = gp.transform(Xk[s:e]) if fitted else gp.fit_transform(Xk[s:e]); fitted = True
        out[keep[s:e]] = res[:, X.shape[1]:]
        if log is not None and k % log_every == 0:
            rate = e / max(time.time() - t0, 1e-9)
            log.write(f'{time.strftime("%H:%M:%S")} edge_id {int(Xk[s, 0])} call {k}/{len(starts)} batch {e - s} '
                      f'{rate:,.0f} rows/s RSS {psutil.Process().memory_info().rss / 1e9:.1f}GB\n'); log.flush()
    return out, fitted, len(starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'bench'])
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--cycle-tw', type=int, default=DAY)
    ap.add_argument('--lc-len', type=int, default=6)
    ap.add_argument('--tag', default='')
    ap.add_argument('--tw', type=int, default=DAY, help='전체 time_window 와 fan/degree 창')
    ap.add_argument('--no-vs', action='store_true', help='vertex stats 끄기')
    ap.add_argument('--start', type=int, default=4_000_000, help='bench: 시작 행')
    ap.add_argument('--rows', type=int, default=600_000, help='bench: 측정 행 수 (워밍업 150만 건 뒤)')
    a = ap.parse_args()

    from snapml import GraphFeaturePreprocessor
    cfg = B.Config(dataset='HI-Large', smoke=False)
    out_dir = cfg.out_dir('gfp')
    chunk_dir = cfg.feat_dir / ('gfp_chunks' + (f'_{a.tag}' if a.tag else '')); chunk_dir.mkdir(exist_ok=True)
    params = gfp_params(a.threads, a.cycle_tw, a.lc_len, a.tw, not a.no_vs); names = gfp_names(params)
    pf = pq.ParquetFile(cfg.feat_dir / 'tx_base.parquet'); cols = ['edge_id', 'src', 'dst', 'timestamp', 'amount_received']
    n_total = pf.metadata.num_rows
    log = open(out_dir / f'gfp_{a.mode}{"_" + a.tag if a.tag else ""}.log', 'a')
    log.write(f'\n==== {time.strftime("%Y-%m-%d %H:%M:%S")} {a.mode} threads={a.threads} tw={a.tw} cycle_tw={a.cycle_tw} lc_len={a.lc_len} vs={not a.no_vs} tag={a.tag!r} 열={len(names)}\n'); log.flush()
    print('로그:', log.name, flush=True)

    gp = GraphFeaturePreprocessor(); gp.set_params(params); fitted = False

    if a.mode == 'bench':
        # 워밍업(창 채우기) 후 a.rows 행의 정상 속도
        wstart = max(0, a.start - WARMUP_ROWS)
        for warm in iter_chunks(pf, cols, wstart, a.start - wstart):
            _, fitted, _ = transform_by_minute(gp, warm, fitted, len(names))
        log.write(f'워밍업 완료 ({a.start - wstart:,} 건)  RSS {psutil.Process().memory_info().rss / 1e9:.1f}GB\n'); log.flush()
        t0 = time.time(); n = 0
        for X in iter_chunks(pf, cols, a.start, a.rows, chunk=min(READ_CHUNK, a.rows)):
            feat, fitted, n_calls = transform_by_minute(gp, X, fitted, len(names), log); n += len(X)
        dt = time.time() - t0
        msg = f'BENCH threads={a.threads} tw={a.tw} cycle_tw={a.cycle_tw} lc_len={a.lc_len} vs={not a.no_vs}: {n:,} 건 {dt:.0f}s → {n / dt:,.0f} 건/s → 전체 {n_total / (n / dt) / 3600:.1f} 시간  RSS {psutil.Process().memory_info().rss / 1e9:.1f}GB'
        log.write(msg + '\n'); print(msg, flush=True); return

    n_parts = (n_total + READ_CHUNK - 1) // READ_CHUNK
    parts = [chunk_dir / f'part_{i:04d}.parquet' for i in range(n_parts)]
    first_missing = next((i for i, p in enumerate(parts) if not p.exists()), n_parts)
    log.write(f'part {n_parts}개, 완료 {first_missing}개\n'); log.flush()
    if first_missing >= n_parts:
        print('이미 완료'); return
    if first_missing > 0:
        wstart = max(0, first_missing * READ_CHUNK - WARMUP_ROWS)
        for warm in iter_chunks(pf, cols, wstart, first_missing * READ_CHUNK - wstart):
            _, fitted, _ = transform_by_minute(gp, warm, fitted, len(names))
        log.write(f'워밍업 완료 ({first_missing * READ_CHUNK - wstart:,} 건)\n'); log.flush()
    t_all = time.time()
    for i, X in enumerate(iter_chunks(pf, cols, first_missing * READ_CHUNK), start=first_missing):
        t0 = time.time()
        feat, fitted, n_calls = transform_by_minute(gp, X, fitted, len(names), log)
        df = pd.DataFrame(feat.astype(np.float16), columns=names); df.insert(0, 'edge_id', X[:, 0].astype(np.int64))
        df.to_parquet(parts[i], index=False, compression='zstd')
        del X, feat, df; gc.collect()
        log.write(f'PART {i:04d}/{n_parts - 1} {time.time() - t0:.0f}s ({n_calls:,} 호출) 누적 {(time.time() - t_all) / 60:.0f}분\n'); log.flush()
    log.write('완료\n'); print('완료', flush=True)


if __name__ == '__main__':
    main()
