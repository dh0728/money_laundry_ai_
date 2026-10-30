# 시험 설계 - binary_classification

## 목적
선행 연구(Multi-GNN · FraudGT · GFP · SALT-GNN)가 쓴 방식을 그대로 재현한 기준선에서 출발해,
**[1단계]** 피처를 하나씩 더하며 **어느 피처가 어느 패턴을 살리는지**, **[2단계]** 모델을 바꿔 가며 **허브(수신 계좌 in-degree 높은 구간)에서 누가 무너지는지** 를 확인하고,
최종 피처·모델로 **[3단계]** 에서 확정된 모델을 **[4단계]** 다운샘플링 데이터와 비교해서 최종적으로 서비스 모델(v10)과 비교한다.


## 공통 설정 (모든 실행 동일)
- **데이터** : IBM AMLworld HI-Large (1.8억 건). 분할은 05 샘플링 실험과 동일 — 학습 2022-08-08~10-02 / 검증 ~10-16 / 테스트 ~10-30, 11월 미사용
- **그래프** : 구간 끝까지의 거래 전부 (검증·테스트 그래프는 이전 거래 포함, Multi-GNN 규칙). `to` / `rev_to` hetero (역방향 MP), ego ID, port numbering
- **학습** : `LinkNeighborLoader` 로 target 2048개가 서브그래프 공유, 이웃 2홉 × 방향별 100, 가중 CE [1, 6.27] (다운샘플링 없음),
  에폭 = 배치 256개 (52만 target), 최대 400 에폭 (약 2바퀴), 검증 4 에폭마다, patience 8, 검증 AP 최고 에폭 선택
- **검증** : 양성 전부 + 정상 2% 가중 표본 (05 와 같은 파일). 테스트: 구간 전체 2,680만 건
- **누수 규칙** : 모든 피처는 거래 시각 기준 과거만. z-norm 은 학습 구간 통계. 노드 피처는 창(window)을 둔다 (누적 금지)
- **메모리 (한도 약 25GB)** : 그래프 도착 노드순 정렬 + `is_sorted=True`, float16/uint8, 그래프는 한 번에 하나만 (`GraphStore.use`)

- **코드** : 공통 로직은 `code/binary_classification/bc_lib.py`. 노트북은 "피처 계산 셀 + bc_lib 호출". 01 은 원본 그대로
- **피처 parquet** : `parquets/HI-Large/features_bc/<단계>.parquet` (edge_id 순). 한 번 계산하면 재사용, pandas 를 거치지 않음
- **결과** : `code/binary_classification/results/HI-Large/<실험명>/` — results.json, compare.csv, *_history.csv, *_test_scores.parquet, analysis_pattern.csv, analysis_degree.csv, curves.png

## 트리 모델 기준선 (모든 단계 공통)
- 매 단계 같은 피처(기본 4 + 그 단계까지의 추가 피처, port 제외)로 LightGBM · XGBoost 를 먼저 학습. scale_pos_weight 6, 검증 AP early stopping
- 역할 두 가지
  1. **피처 자체의 신호 확인**: 트리에서도 오르면 피처가 정보를 갖는 것. 트리는 안 오르는데 GNN 만 오르면 "이웃 구조와 결합해야 의미 있는 피처"
  2. **벤치마크**: 
    - 트리와 GNN에 똑같은 피처를 넣었을 때 둘의 점수 차이는 피처 때문이 아니라 GNN이 이웃 거래까지 본 것 때문이라는 것을 확인
    - GFP 논문의 그래프 패턴을 미리 피처로 계산해서 트리에 넣으면 GNN 만큼 나온다(GFP+XGBoost ≈ GNN) 주장을 우리 데이터에서 검증한다. 
- 트리 점수 parquet 도 저장해 패턴별 recall · degree 구간 표에 GNN 과 나란히 둔다
- 5단계(GFP 모티프)까지 가면 트리 = GFP 논문 구성 그대로 → HI-Large GFP+LightGBM 58.0 (Table 4) 과 직접 비교 가능

## 모든 실행이 내는 표 (0단계 분석 함수)
1. **전체 지표** (검증 / 테스트): AP, F1(최적 임계값), F1@0.5(논문 비교용), precision, recall, recall@1%, unmatched recall@1%, ECE
2. **패턴 유형별 recall@1%** — `patterns.parquet` 조인. 임계값은 테스트 전체 상위 1% (알림 예산) 하나로 고정.
   `FAN-OUT` · `FAN-IN` · `GATHER-SCATTER` · `SCATTER-GATHER` · `CYCLE` · `RANDOM` · `BIPARTITE` · `STACK` · `UNMATCHED(패턴 파일에 없는 양성)`
   → "이 피처를 넣었더니 노린 패턴이 올랐나" 를 보는 표. 이전 단계와 diff 로 비교
3. **수신 계좌 in-degree 구간별** AP · F1 · F1@0.5 · recall@1% — SALT-GNN 프로토콜. 구간 [1-4] [5-9] [10-19] [20-49] [50-99] [100+],
   in-degree 는 테스트 구간 끝까지 거래에서 센 "받은 거래 수". 학습·선택은 그대로, 테스트 예측만 구간으로 나눈다 (사후 분석)
   → "허브에서 무너지나" 를 보는 표

- 단계 간 비교는 같은 검증 표본·같은 테스트 전체·같은 지표라 그대로 나란히 놓으면 된다

## 실행 순서 (각 단계 공통)
1. 새 피처를 DuckDB 창 함수로 계산해 parquet 저장 → 평균값 / 계좌 하나 타임라인 / 라벨별 평균으로 값이 의도대로 나왔는지 확인
2. 이전 단계 피처 + 새 피처로 z-norm
3. 트리 모델 (LightGBM, XGBoost, scale_pos_weight 6) — 피처 자체의 신호 확인 + 피처 중요도. 트리는 port 없이 (GFP 의 "basic" 과 맞춤)
4. GINe 학습 (01 과 같은 설정) → 테스트 전체 → 표 1~3
5. 이전 단계와 비교: 전체 AP, 패턴별 recall diff, degree 구간별 AP
6. 가설대로 노린 패턴이 올랐으면 다음 단계로. 안 올랐으면 피처 정의(창 길이, 집계 방식)를 한 번 바꿔 보고, 그래도 아니면 기록하고 다음 단계로 (피처는 누적해서 가져감)
- 새 노트북은 먼저 `HI-Small + smoke=True` 로 끝까지 한 번 돌려 에러만 잡고 Large 로 간다 (Large 는 단계당 약 1일: 학습 18~20시간 + 테스트 3시간)

---


## [1단계] 피처별 실험을 통해 검증(HI-Large, GINe 고정)
- 모델을 GINe 하나로 고정하는 이유: 피처 효과를 모델 차이와 섞지 않기 위해. 
- Large 에서 하는 이유: 패턴 수·시간 창·허브 분포가 운용 규모와 같아야 "이 피처가 이 패턴을 살렸다" 가 의미 있음 (Small 은 10일짜리라 장기 패턴이 적음)

### 01 기본 4 + port 2 - 완료
- 피처: Timestamp, Amount Received, Received Currency, Payment Format (Multi-GNN 기본) + in_port, out_port (Multi-GNN ports)
- 역할: 선행 연구 재현 기준선. 노리는 패턴 없음
- 결과 (검증): GINe AP 0.461 (best epoch 108), GINe+EU 0.414 (best 76). 05 random_r10 (v10, 36피처) 0.466 과 거의 같음
- 이후 단계는 GINe 만 (EU 는 01 에서 더 낮았고 25% 느림. 2단계 모델 비교 때 다시 봄)

### 02 같은 쌍 반복 (pair history)
- 노리는 패턴: STACK, BIPARTITE (같은 상대 반복 송금, 왕복). FAN · CYCLE 은 거의 안 변해야 정상
- 피처 (v10 엣지 피처에서 가져옴, 전부 ln 변환 후 z-norm):
  - `pair_prior_count`: 이 거래 전에 A→B 로 보낸 횟수 (처음부터, 같은 분 제외)
  - `pair_gap_minutes`: 직전 A→B 거래까지의 분
  - `pair_count_1h`: 최근 60분 안 A→B 횟수
  - `reverse_history_count`: 반대 방향 B→A 누적 횟수 (전날까지, 일 단위 ASOF)
- 계산: DuckDB 창 함수 `PARTITION BY src, dst ORDER BY ts RANGE ... INTERVAL '1 minute' PRECEDING` (같은 분 제외는 서비스와 동일)
- 노트북: `02_pair_history.ipynb`

### 03 계좌 단기 활동 (account activity)
- 노리는 패턴: FAN-IN, FAN-OUT, GATHER-SCATTER (한 계좌가 짧은 시간에 여러 상대와)
- 피처 (src · dst 각각, 창 1h 와 1일): 
  - `src_out_1h`, `src_out_1d` :	송신 계좌가 최근 1h / 24h에 보낸 건수
  - `src_uniq_dst_1d` : 송신 계좌의 최근 24h 상대 계좌 수
  - `src_out_amt_1d` :	송신 계좌의 최근 24h 송금액 합
  - `dst_in_1h`, `dst_in_1d`, `dst_uniq_src_1d`, `dst_in_amt_1d` : 수신 계좌 쪽 같은 네 개
  - **상대 계좌 수** (distinct) — v10 의 `sender_count_1h` 는 거래 수라 "같은 상대 30번" 과 "30곳 1번씩" 을 구분 못 함. GFP fan 이 가장 큰 효과였던 이유
  - 금액 합
- 창 기준: 거래 시각 이전만. 누적 버전은 넣지 않음 (04_node_count_drift 에서 드리프트 확인됨)

### 04 계좌 금액 분포 (account amount stats)
- 노리는 패턴: 스머핑(작게 쪼개 반복), 허브 구간 구별 (SALT C1 — 평균 뒤에 숨는 이상치)
- 피처 (src · dst 각각, 창 1일): 받은/보낸 금액의 mean, std, max + 이 거래 금액의 z (계좌 기준)
- 근거: GFP ablation 에서 계좌 통계 +4~8p, SHAP 2위 "도착 계좌 받은 금액 합"

### 05 GFP 모티프 (multi-hop motifs)
- 노리는 패턴: CYCLE, SCATTER-GATHER (2홉 GNN 이 구조적으로 못 세는 것)
- 피처: snapml `GraphFeaturePreprocessor` — 사이클 길이별(≤10) 참여 수, scatter-gather 중간 노드 수별 참여 수. 창은 패턴 시간 폭을 재서 정함 (GFP 의 1일·6시간은 처리량 타협값)
- 계산: 1.8억 건을 시각순 배치로 흘려 parquet 에 이어 쓰기 (별도 노트북). 비용이 커서 마지막
- 기대는 낮게: GFP·BlazingAML 모두 트리 모델에서 +1~4p. GNN 에서 다를지 확인

### 1단계 산출물
- 단계별 전체 AP / 패턴별 recall / degree 구간 표를 한 표로 누적 (`docs/binary_classification/feature_ladder.md`)
- 최종 피처 세트 = 패턴 recall 을 올린 단계만 누적한 것

---

## [2단계] 모델 비교 (HI-Small, 최종 피처 고정, 시드 3)
- 목적: 같은 피처에서 모델에 따라 전체 성능과 **허브 구간** 성능이 어떻게 다른가. "허브 계좌를 못 찾는 모델" 을 가려냄
- 모델: GINE, GINE+EU, PNA, GAT, TransConv (= v10 백본), FraudGT, SALT-GAT, SALT-Trans
  - SALT는 수식이 단순해 직접 구현 (PNA 집계 + attention 을 층마다 concat → MLP → residual)
  - 하이퍼파라미터: 모델별로 새로 튜닝하지 않고 논문 값을 그대로 쓴다. 튜닝 정도가 다르면 모델 차이인지 튜닝 차이인지 못 가리고, 논문 수치와 조건을 맞추기 위해
    | 모델 | lr | hidden | 층 | 양성 가중치 | dropout / final | 출처 |
    |---|---|---|---|---|---|---|
    | GINE, GINE+EU | 0.0062 | 66 | 2 | 6.27 | 0.010 / 0.105 | Multi-GNN model_settings 'gin' |
    | PNA | 0.0006 | 20 | 2 | 7.08 | 0.083 / 0.288 | Multi-GNN 'pna' |
    | GAT | 0.006 | 64 (heads 4) | 2 | 6 | 0.009 / 0.1 | Multi-GNN 'gat' |
    | TransConv | 0.001 | 64 (heads 4) | 2 | 6 | 0.009 / 0.1 | SALT Appendix I |
    | FraudGT | 0.001, AdamW + cosine (warmup 5) | 64 (heads 8) | 2 + post 2 | 6 | 0.2 / attn 0.3 | FraudGT configs · SALT Appendix I |
    | SALT-Trans | 0.0012 | 20 | 2 | 7 | 0.083 / 0.288 | SALT Appendix I (PNA 값 상속) |
    | SALT-GAT | 0.0006 | 20 | 2 | 7 | 0.083 / 0.288 | SALT Appendix I |
    - 공통: Adam (FraudGT 만 AdamW), 배치 2048, 이웃 방향별 100, z-norm, 검증 AP 로 에폭 선택
    - 논문 값은 Small/Medium 기준이라 우리 피처·분할에서 최적은 아닐 수 있음. 2단계도 Small 이므로 그대로 두고, 3단계(Large) 에서 상위 2개만 lr·hidden 한 번 점검
  
- Small 을 쓰는 이유: SALT 논문이 Small/Medium 에서 했으니 논문 수치와 직접 비교 가능, 모델당 1~2시간이라 7모델 × 3시드 ≈ 2~3일
- 분할: Small 은 10일짜리라 일 단위 60/20/20 (Multi-GNN 규칙)
- 평가: 표 1~3 + 시드 3개 평균±표준편차. 전체 F1 과 deg[50-99]·[100+] 구간 F1 을 같이 보고
- 선정: 전체·허브 구간 모두 상위인 2개

## [3단계] 확정 (HI-Large) — 2일
- 2단계 상위 2개 × 1단계 최종 피처 → Large 에서 각 1일 → 최종 모델 1개

## [4단계] 샘플링 (HI-Large, 최종 모델)
- 01 구조에서는 정상 전부 + 가중 손실로 비용 문제가 풀렸으므로, 남는 질문은 "줄이면 성능이 더 오르나" (양성 반복 노출)
- 조건: none (현재) vs random r=10 vs r=30 vs hard-negative r=10, π_eff 0.25. **같은 학습 예산**(본 target 수) 으로 맞춤
- 구현: 학습 target 목록과 손실 가중치만 바꿈 (`make_loader` 의 targets + π_eff). 05 처럼 별도 파이프라인 불필요
- 05 결과(v10 구조)와 나란히 → "구조가 바뀌어도 샘플링 결론이 유지되나"

## [5단계] 서비스 반영
- 최종 피처 → `materials.py` / `preprocess.py`, 최종 모델 → 추론서버. 공유 서브그래프 방식이면 추론 구조도 바뀌므로 별도 설계

## 일정 (GPU 1대 기준)
| 단계 | 기간 |
|---|---|
| 0 분석 도구 | 반나절 (완료) |
| 1 피처 사다리 02~05 + v10 참고 | 5일 |
| 2 모델 비교 | 2~3일 |
| 3 확정 | 2일 |
| 4 샘플링 | 3일 |
| 합계 | 약 2주 |