# 02 노트북 "결과" 절 (cell 35 ~ 끝) 을 통째로 바꾼다. 저장된 점수 parquet 만으로 돌아가게.
import json, sys

MD = lambda s: dict(cell_type='markdown', metadata={}, source=s)
CODE = lambda s: dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=s)

cells = [
MD("""# 결과 — 저장된 점수 파일만으로 돌아간다
커널을 새로 켰으면 **0 절(설정 셀 3개)까지만** 돌리고 바로 여기부터. 재료는 `results/<DATASET>/<RUN_NAME>/`
- `{model}_test_scores.parquet` (edge_id, y, score) — 7 절·4 절에서 저장
- `results.json` — 검증/테스트 지표 (`results` dict 복구용)
- dst 와 수신자 in-degree 만 DuckDB 에서 다시 읽는다 (1~2 분)

운영점(컷)은 두 종류: **예산 컷** `r@b%` = 테스트 전체 상위 b% 를 알림으로 (0.05 / 0.1 / 0.5 / 1 / 2 / 5 / 10 % / all),
**argmax** = 확률 0.5 (논문 F1 운영점, Multi-GNN 부록 Table 13 의 패턴별 recall 과 같은 기준). 0.1% ≈ 양성 수 규모 ≈ argmax 와 비슷한 알림 수."""),

MD("## 8. 준비 — 점수 · 라벨 · in-degree"),
CODE("""import importlib; importlib.reload(B)
MODELS = [n for n in ['lightgbm', 'xgboost', 'gine'] if (OUT_DIR / f'{n}_test_scores.parquet').exists()]
scores = {n: B.load_scores(OUT_DIR / f'{n}_test_scores.parquet') for n in MODELS}
if 'test_df' not in globals() or 'in_degree' not in globals():
    con = L.connect(threads=8, memory='12GB')
    test_df = B.test_full(con, cfg.test_window); in_degree = B.recipient_indegree(con, cfg); con.close()
for n in MODELS:
    assert (scores[n]['edge_id'].to_numpy() == test_df['edge_id'].to_numpy()).all(), f'{n} 점수와 라벨 순서 불일치'
if 'results' not in globals() or not results:
    results = json.load(open(OUT_DIR / 'results.json'))['results']
print(MODELS, '| 테스트 간선', f"{len(test_df):,}", '| 양성', f"{int(test_df.y.sum()):,}", '| 컷', list(B.cut_thresholds(scores[MODELS[0]]['score'].to_numpy())))"""),

MD("""## 9. 운영점별 precision / recall
같은 점수를 컷만 바꿔 가며 본다. 예산을 줄이면 precision 이 오르고 recall 이 떨어지는 그 교환을 한 표에."""),
CODE("""# 컷마다 임계값 · 알림 수 · TP/FP/FN · precision · recall (모델별). 'all' = 전부 알림 (precision = 양성 비율)
cut_df = pd.concat([B.cut_table(scores[n]).assign(model=n) for n in MODELS])
cut_df.to_csv(OUT_DIR / 'analysis_cuts.csv', encoding='utf-8-sig')
cut_df[cut_df.model == 'gine']"""),
CODE("""# 모델 나란히 — 컷별 recall / precision
cut_df.reset_index().pivot(index='cut', columns='model', values=['recall', 'precision']).loc[cut_df.index.unique()]"""),
CODE("""# PR 곡선 — 컷을 연속으로 훑은 것. AP = 곡선 아래 면적. 세로선 = GINe argmax 운영점
B.plot_pr(scores, path=OUT_DIR / 'pr_curve.png', mark=(cut_df.loc['argmax'].query("model == 'gine'")['recall'].iloc[0], 'GINe argmax'))"""),
CODE("""# 혼동행렬 — GINe, 컷 4개 나란히. 색 = 행 비율 (세탁 행 오른쪽 칸 = recall), 숫자 = 건수
B.plot_confusion_grid(scores['gine'], cuts=('r@0.1%', 'r@1%', 'r@10%', 'argmax'), title='GINe (pair_history)', path=OUT_DIR / 'confusion_cuts.png')"""),

MD("""## 10. 패턴별 — 어느 패턴을 어느 컷에서 놓치나
패턴 안의 거래는 모두 양성이라 recall 만 본다. 예산을 줄일 때 먼저 떨어지는 패턴이 "모델이 덜 확신하는" 패턴."""),
CODE("""# 패턴 × 컷 recall (모델별 저장, GINe 표시). 'recall' 열 = 1% (01 과 호환), mean_score = 그 패턴 양성의 평균 점수
pattern_df = pd.concat([B.pattern_recall(scores[n]).assign(model=n) for n in MODELS])
pattern_df.to_csv(OUT_DIR / 'analysis_pattern.csv', encoding='utf-8-sig')
CUT_COLS = [c for c in pattern_df.columns if c.startswith('r@') or c in ('all', 'argmax')]
pattern_df[pattern_df.model == 'gine'][['n'] + CUT_COLS + ['mean_score']]"""),
CODE("""# 같은 표를 그림으로 — 행 = 패턴, 열 = 컷. 왼쪽(빡빡한 예산)에서 먼저 하얘지는 행이 약한 패턴
B.plot_heatmap(pattern_df[pattern_df.model == 'gine'][CUT_COLS], title='GINe recall: pattern x cut', path=OUT_DIR / 'pattern_x_cut.png')"""),
CODE("""# 놓친 세탁(FN) 의 구성 — 컷마다 FN 이 어느 패턴에서 나오나. FN% 가 pos_share 보다 크면 그 패턴을 유독 놓침
groups = dict(pattern=B.pattern_array(scores['gine']), degree_bin=B.degree_bin_of(test_df['dst'].to_numpy(), in_degree))
cb = B.confusion_breakdown(scores['gine'], groups)
for axis in groups:
    cb['fn'][axis].to_csv(OUT_DIR / f'analysis_fn_{axis}.csv', encoding='utf-8-sig'); cb['fp'][axis].to_csv(OUT_DIR / f'analysis_fp_{axis}.csv', encoding='utf-8-sig')
cb['fn']['pattern'][['pos_share'] + [c for c in cb['fn']['pattern'].columns if c.startswith('FN% ')]]"""),

MD("""## 11. 수신자 in-degree 층화 (SALT-GNN 프로토콜)
허브(차수 큰 수신자) 쪽 거래가 약점인지. AP/F1 은 구간 안에서 따로 계산, recall/precision 은 전체 컷 임계값을 구간에 적용."""),
CODE("""# 구간별 AP · F1 · 양성 비율 (모델별 저장, GINe 표시)
degree_df = pd.concat([B.degree_stratified(scores[n], test_df['dst'].to_numpy(), in_degree).assign(model=n) for n in MODELS])
degree_df.to_csv(OUT_DIR / 'analysis_degree.csv', encoding='utf-8-sig')
degree_df[degree_df.model == 'gine']"""),
CODE("""# 구간 × 컷 — recall 과 precision (GINe). 같은 임계값인데 구간마다 precision 이 다르면 알림이 그쪽에 몰리는 것
deg_cut = B.group_by_cut(scores['gine'], groups['degree_bin'])
order = [b for b in [f'[{lo}-{hi}]' if hi < 10**9 else f'[{lo}+]' for lo, hi in B.DEGREE_BINS] if b in deg_cut['recall'].index]
deg_cut['recall'].loc[order].to_csv(OUT_DIR / 'analysis_degree_x_cut_recall.csv', encoding='utf-8-sig')
deg_cut['precision'].loc[order].to_csv(OUT_DIR / 'analysis_degree_x_cut_precision.csv', encoding='utf-8-sig')
display(deg_cut['recall'].loc[order]); deg_cut['precision'].loc[order]"""),
CODE("""# 잘못 잡은 정상(FP) 이 어느 차수 구간에 몰리나 — FP% 가 neg_share 보다 크면 그 구간에 알림이 과함
cb['fp']['degree_bin'][['neg_share'] + [c for c in cb['fp']['degree_bin'].columns if c.startswith('FP% ')]]"""),

MD("""## 12. 01 (basic_features) 과 비교
01 쪽 표는 `analysis_01.ipynb` 가 만든다. 없으면 건너뛴다."""),
CODE("""# 패턴별 recall — 1% 와 argmax 두 컷에서 01 GINe basic vs 이번 GINe
prev_path = Path.cwd() / 'results' / cfg.dataset / PREV_RUN / 'analysis_pattern.csv'
if prev_path.exists():
    prev = pd.read_csv(prev_path, index_col=0); prev = prev[prev.model == 'gine_basic']; cur = pattern_df[pattern_df.model == 'gine']
    cols = {'01 r@1%': prev['recall'], '02 r@1%': cur['recall']}
    if 'argmax' in prev.columns:
        cols.update({'01 argmax': prev['argmax'], '02 argmax': cur['argmax']})
    cmp = pd.DataFrame(cols); cmp['diff r@1%'] = cmp['02 r@1%'] - cmp['01 r@1%']
    if 'argmax' in prev.columns: cmp['diff argmax'] = cmp['02 argmax'] - cmp['01 argmax']
    display(cmp.loc[[p for p in B.PATTERN_ORDER + ['ALL'] if p in cmp.index]].round(4))
else:
    print('01 분석표 없음 — analysis_01.ipynb 를 먼저 돌리면 비교가 나온다:', prev_path)"""),
CODE("""# 수신 계좌 degree 구간별 AP — 허브 구간이 달라졌나
prev_path = Path.cwd() / 'results' / cfg.dataset / PREV_RUN / 'analysis_degree.csv'
if prev_path.exists():
    prev_d = pd.read_csv(prev_path, index_col=0); prev_d = prev_d[prev_d.model == 'gine_basic']; cur_d = degree_df[degree_df.model == 'gine']
    display(pd.DataFrame({'01 AP': prev_d['AP'], '02 AP': cur_d['AP'], 'diff': cur_d['AP'] - prev_d['AP'],
                          '01 F1': prev_d['F1'], '02 F1': cur_d['F1'], 'n_pos': cur_d['n_pos']}).round(4))
else:
    print('01 분석표 없음:', prev_path)"""),

MD("## 13. 비교표 + 저장"),
CODE("""compare_df = B.save_results(OUT_DIR, cfg, results, features=dict(tree=TREE_FEATURES, gnn=GNN_EDGE_FEATURES))
compare_df"""),
CODE("""# 01 전체 지표와 나란히
prev_c = pd.read_csv(Path.cwd() / 'results' / cfg.dataset / PREV_RUN / 'compare.csv', index_col=[0, 1])
pd.concat([prev_c.assign(run=PREV_RUN), compare_df.assign(run=RUN_NAME)]).reset_index().set_index(['run', 'model', 'split'])[['AP', 'F1', 'F1_at_0.5', 'recall_at_1pct', 'unmatched_recall_at_1pct']]"""),
CODE("""sorted(p.name for p in OUT_DIR.iterdir())   # 이 폴더가 report.md 의 재료"""),
]

if __name__ == '__main__':
    path, start = sys.argv[1], int(sys.argv[2])
    nb = json.load(open(path, encoding='utf-8'))
    tail = nb['cells'][start:]
    keep_next = [c for c in tail if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## 10. 다음')]
    if keep_next:
        src = ''.join(keep_next[0]['source']).replace('## 10. 다음', '## 14. 다음')
        keep_next = [MD(src)]
    nb['cells'] = nb['cells'][:start] + cells + keep_next
    json.dump(nb, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('cells:', len(nb['cells']))
