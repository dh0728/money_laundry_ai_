"""서비스 이진 모델(multi_fraudgt_hgb)로 실험 데이터의 거래에 점수를 매기는 도우미. 00_service_scores.ipynb 가 쓴다.

모델 코드와 가중치는 서비스 저장소(money_laundry/backend/inference-server)의 것을 그대로 불러 쓴다.
여기 있는 것은 그 패키지를 찾아 불러오는 일과, 서비스의 하루 실행과 같은 방식으로 하루치 거래를 채점하는 일뿐이다.

주의: 이 모듈은 inference-server 폴더를 sys.path 에 넣는다. gen_lib.load_generator 가 넣는 worker 폴더에도
이름이 같은 패키지(models, common)가 있어서, 한 커널에서 둘을 같이 불러오면 안 된다.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PACKAGE = Path('models') / 'multi_fraudgt_hgb'


def q(value):
    """SQL 문자열 리터럴"""
    return "'" + str(value).replace("'", "''") + "'"


# ============================================================================ 1. 서비스 패키지
def find_server(start=None):
    """나란히 있는 money_laundry 저장소에서 추론 서버 폴더를 찾는다. 이진 모델 가중치가 들어 있어야 한다."""
    here = Path(start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        server = p / 'money_laundry' / 'backend' / 'inference-server'
        if (server / PACKAGE / 'models' / 'hgb.joblib').is_file():
            return server
        if server.is_dir():
            raise FileNotFoundError(f'{server} 에 {PACKAGE} 가중치가 없다. money_laundry 를 dev 브랜치 최신으로 맞춰야 한다')
    raise FileNotFoundError('money_laundry 저장소를 못 찾음 (이 저장소와 나란히 있어야 한다)')


def git_commit(path):
    """그 폴더가 든 저장소의 현재 커밋 한 줄"""
    try:
        return subprocess.run(['git', '-C', str(path), 'log', '-1', '--format=%h %cs %s'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return '(git 기록을 못 읽음)'


def versions(server):
    """서비스가 고정한 라이브러리 버전(requirements-models.txt)과 지금 환경의 버전을 나란히 본다.
    scikit-learn 이 다르면 트리(hgb.joblib)가 안 열리거나 점수가 달라질 수 있다."""
    text = (Path(server) / 'requirements-models.txt').read_text(encoding='utf-8')
    wanted = dict(re.findall(r'^([A-Za-z0-9_\-]+)==(\S+)\s*$', text, re.M))
    for name in ('torch', 'pyg_lib'):                    # 이 둘은 설치 안내 주석에만 적혀 있다
        found = re.search(rf'{name}==([0-9][^\s]*)', text)
        if found:
            wanted[name] = found.group(1)
    rows = []
    for name, want in wanted.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = '(없음)'
        rows.append(dict(패키지=name, 서비스=want, 여기=have, 같음=have.split('+')[0] == want))
    return pd.DataFrame(rows).sort_values('패키지').reset_index(drop=True)


def worker_settings(server):
    """의심 기준과 씨앗 보강 값. 백엔드 worker 쪽 spec.py 에 적힌 값을 글자로 읽는다(불러오면 패키지 이름이 겹친다)."""
    spec = Path(server).parent / 'api-server' / 'worker' / PACKAGE / 'spec.py'
    text = spec.read_text(encoding='utf-8')
    out = {}
    for key, name in [('threshold', 'RECOMMENDED_THRESHOLD'), ('seed_low', 'ALERT_SEED_LOW'), ('chain_floor', 'ALERT_CHAIN_FLOOR')]:
        found = re.search(rf'^{name}\s*=\s*([0-9.eE+\-]+)', text, re.M)
        assert found, f'{spec} 에 {name} 이 없다'
        out[key] = float(found.group(1))
    return out


def load_model(server, device='cpu', threads=2):
    """서비스 패키지를 불러오고 모델(그래프 모델 + 트리 + 결합)을 메모리에 올린다. 가중치 파일의 체크섬도 여기서 확인된다."""
    server = Path(server)
    loaded = sys.modules.get('models')
    if loaded is not None and str(server) not in str(getattr(loaded, '__file__', '') or list(getattr(loaded, '__path__', ['']))[0]):
        raise RuntimeError('다른 곳의 models 패키지가 이미 불려 있다(gen_lib 의 생성기와 같은 커널). 커널을 다시 시작하고 이 노트북만 돌린다')
    if str(server) not in sys.path:
        sys.path.insert(0, str(server))
    from models.multi_fraudgt_hgb import spec
    from models.multi_fraudgt_hgb.backend_model.base_features import KYC_COLUMNS, TX_COLUMNS
    from models.multi_fraudgt_hgb.backend_model.inference import Predictor
    from models.multi_fraudgt_hgb.backend_model.preprocessing import prepare
    from models.multi_fraudgt_hgb.backend_model.schema import ENTITY_TYPES
    return SimpleNamespace(server=server, spec=spec, prepare=prepare, predictor=Predictor(device, threads),
                           tx_columns=list(TX_COLUMNS), kyc_columns=list(KYC_COLUMNS), entity_types=list(ENTITY_TYPES))


# ============================================================================ 2. 하루 채점
def score_day(M, transactions, kyc, day, work_dir, threads=2, memory='2GB', progress=True):
    """day 하루의 거래를 그날 마감 기준으로 채점한다. 서비스의 하루 실행과 같은 호출이다.

    - 피처: 처음부터 day 까지의 거래 전체로 다시 만든다 (context_end = day). day 뒤의 거래는 읽지 않는다
    - 대상: day 의 거래 전부 (target_start = target_end = day)
    - 배치 크기와 샘플링 시드는 서비스 값(spec.BATCH_SIZE, spec.SAMPLING_SEED). 점수가 이 둘에 따라 달라진다
    반환: (DataFrame[edge_id, score, gnn_score, hgb_score], 걸린 시간 dict)
    """
    day = str(pd.Timestamp(day).date())
    run = Path(work_dir) / f'run_{day}'
    shutil.rmtree(run, ignore_errors=True)               # 패키지가 빈 폴더를 요구한다
    settings = dict(transactions=str(transactions), kyc=str(kyc), context_end=day, target_start=day, target_end=day,
                    output_dir=str(run), threads=int(threads), memory_limit=str(memory))
    try:
        t0 = time.time()
        prepared = M.prepare(settings, progress=False)
        t1 = time.time()
        out = M.predictor.predict(prepared, batch_size=M.spec.BATCH_SIZE, seed=M.spec.SAMPLING_SEED, progress=progress)
        t2 = time.time()
    finally:
        shutil.rmtree(run, ignore_errors=True)           # 작업 DB 가 크다. 하루 끝나면 지운다
    scores = out[['edge_id', 'risk_score', 'gnn_score', 'hgb_score']].rename(columns={'risk_score': 'score'})
    scores = scores.astype({'edge_id': 'int64', 'score': 'float64', 'gnn_score': 'float32', 'hgb_score': 'float32'})
    return scores.sort_values('edge_id').reset_index(drop=True), dict(전처리_초=round(t1 - t0, 1), 추론_초=round(t2 - t1, 1))


def self_check(M, work_dir, threads=2, memory='2GB'):
    """패키지에 딸린 예제(거래 81건, 대상 2022-10-03 의 18건)를 채점해 패키지가 적어 둔 점수와 비교한다.
    서비스 테스트의 허용 오차는 절대값 2e-6 이다(CPU 기준). GPU 에서는 조금 더 벌어질 수 있다."""
    folder = M.server / 'tests' / 'fixtures' / 'multi_fraudgt_hgb'
    expected = pd.DataFrame([json.loads(line) for line in (folder / 'expected_predictions.jsonl').read_text(encoding='utf-8').splitlines()])
    scores, _ = score_day(M, folder / 'transactions.parquet', folder / 'kyc.parquet', '2022-10-03', work_dir, threads, memory, progress=False)
    both = expected.merge(scores, on='edge_id', how='outer', suffixes=('_기대', ''), indicator=True)
    diff = (both.score - both.risk_score).abs()
    return dict(기대_건수=len(expected), 채점_건수=len(scores), 짝이_안_맞는_거래=int((both._merge != 'both').sum()),
                최대_차이=float(diff.max()), 허용_오차_2e_6_안=bool(diff.max() <= 2e-6))


# ============================================================================ 3. 입력 준비
def daily_counts(con, transactions):
    """날짜별 거래 수와 세탁 거래 수"""
    return con.execute(f'''SELECT CAST(ts AS DATE) AS day, count(*) AS 거래, count(*) FILTER (WHERE is_laundering) AS 세탁
                           FROM read_parquet({q(transactions)}) GROUP BY 1 ORDER BY 1''').df()


def propose_window(days, warmup_days, floor=0.1):
    """평가 구간 제안.
    - 시작: 데이터 첫날 + warmup_days. 앞쪽 며칠은 계좌·상대 이력이 안 쌓여 피처가 비어 있다
    - 끝: 거래 수가 (가장 많은 날 × floor) 이상인 마지막 날. 그 뒤는 정상 거래가 끊기고 세탁 거래만 남는 꼬리다"""
    day = pd.to_datetime(days.day)
    start = day.min() + pd.Timedelta(days=warmup_days)
    busy = day[days.거래.to_numpy() >= floor * days.거래.max()]
    end = busy.max()
    assert start <= end, f'평가 구간이 비었다: 시작 {start.date()} > 끝 {end.date()}'
    return str(start.date()), str(end.date())


def write_inputs(con, transactions, kyc, out_dir, M, end_day):
    """모델이 읽을 두 파일을 쓴다. 서비스의 materials.normalize 와 같은 규칙이다.
    - 거래: 모델이 쓰는 열만, end_day 까지만. 정답 열과 평가 구간 뒤 거래는 넣지 않는다
    - 계좌: 소유주 유형이 학습한 6종이 아니면 비운다(모델이 UNKNOWN 으로 읽는다). 국가가 없으면 비운다"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tx_out, kyc_out = out_dir / 'tx_model.parquet', out_dir / 'kyc_model.parquet'
    columns = ', '.join(M.tx_columns)
    con.execute(f'''COPY (SELECT {columns} FROM read_parquet({q(transactions)}) WHERE CAST(ts AS DATE) <= DATE {q(end_day)} ORDER BY edge_id)
                    TO {q(tx_out)} (FORMAT PARQUET, COMPRESSION ZSTD)''')
    types = ', '.join(q(name) for name in M.entity_types)
    con.execute(f'''COPY (SELECT node_id, CAST(entity_id AS VARCHAR) AS entity_id,
                                 CASE WHEN entity_type IN ({types}) THEN entity_type END AS entity_type,
                                 nullif(trim(country), '') AS country
                          FROM read_parquet({q(kyc)}) ORDER BY node_id)
                    TO {q(kyc_out)} (FORMAT PARQUET, COMPRESSION ZSTD)''')
    info = con.execute(f'''SELECT (SELECT count(*) FROM read_parquet({q(tx_out)})) AS 거래,
                                  (SELECT count(*) FROM read_parquet({q(kyc_out)})) AS 계좌,
                                  (SELECT count(*) FROM read_parquet({q(kyc)}) WHERE entity_type IS NOT NULL AND entity_type NOT IN ({types})) AS 유형을_비운_계좌,
                                  (SELECT count(*) FROM read_parquet({q(kyc_out)}) WHERE entity_type IS NULL) AS 유형_없는_계좌,
                                  (SELECT count(*) FROM read_parquet({q(kyc_out)}) WHERE country IS NULL) AS 국가_없는_계좌''').df().iloc[0].to_dict()
    return tx_out, kyc_out, {k: int(v) for k, v in info.items()}


# ============================================================================ 4. 지표
def at_threshold(name, y, score, threshold):
    """기준값 이상을 고를 때의 건수와 정밀도·재현율"""
    y = np.asarray(y).astype(bool)
    hit = np.asarray(score) >= threshold
    tp, fp, fn = int((hit & y).sum()), int((hit & ~y).sum()), int((~hit & y).sum())
    return dict(구분=name, 거래=len(y), 세탁=int(y.sum()), 고른_거래=tp + fp, 맞은_세탁=tp, 정상=fp,
                정밀도=round(tp / max(tp + fp, 1), 4), 재현율=round(tp / max(tp + fn, 1), 4))
