Multi-GNN / FraudGT / SALT (GNN)

엣지피처
- 거래 자체 값(시각·금액·통화·방식) + port, 창개념 없음, 거래 하나의 속성이니까

노드피처
- 없음(전부 1 + ego)
- 계좌 정보는 메시지 패싱이 이웃에서 직접 모음

GFP / Altman 트리
엣지피처
- 거래가 속한 패턴수 (사이클·SG) - 창 안에서 센 것
- fan-in/out·degree·금액 통계 - 창 안에서 (6시간·1일), 거래 행에 복사

# 시험 설계

## 1단계 피처별 실험을 통해 검증(HI-Large, GINe 고정)

### 01 기본 4 + port 
- 시작점으로 삼기위한 단계
- 노리는 패턴은 X 

### 02 같은 쌍 반복 (pair 이력·간격·1h 횟수)
- 노리는 패턴 : `STACK`, `BIPARTITE`

#### 02에서 넣을 피처
- `pair_prior_count` : 이 거래 전에 A→B로 보낸 적이 몇 번인가 (처음부터 지금까지)
- `pair_gap_minutes` : 직전 A→B 거래가 몇 분 전이었나
- `pair_count_1h` : 최근 1시간 안에 A→B가 몇 번 있었나
- `reverse_history_count` : 반대 방향 B→A는 몇 번 있었나 (처음부터 지금까지)


### 03 계좌 단기 활동, 창 1h·1일 (거래 수·상대 수·금액 합)
- 노리는 패턴 : `FAN-IN/OUT`, `GATHER-SCATTER`

### 04 계좌 금액 분포, 창 1일 (mean/std/max)
- 노리는 패턴 : 스머핑, 허브 구별

### 05 GFP 모티프 (사이클·SG 참여 수)
- 노리는 패턴 : `CYCLE`, `SCATTER-GATHER`


### 2