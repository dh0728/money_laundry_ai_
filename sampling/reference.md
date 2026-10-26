## 선행 연구의 클래스 불균형 처리

### 1. Realistic Synthetic Financial Transactions for Anti-Money Laundering Models (Altman et al., NeurIPS 2023)
- AMLworld 데이터셋 원 논문
- 데이터셋 : 본문은 HI·LI × Small, Medium (GNN·트리 모델 모두)
  보충자료에서 HI·LI × Large를 트리 모델(GFP+XGBoost, LightGBM)로만 추가 실험. GNN의 Large 결과는 없음
- 분할 : 시간순 60/20/20. 검증·테스트 그래프에 이전 거래 포함
- 방식 :
  - GNN - 가중 손실 (minority class weight 튜닝, 가중치 범위 6~8)
  - 트리 모델 - scale_pos_weight (가중치 범위 1~10)
  - 다운샘플링·오버샘플링 없음. 샘플링은 규모 때문에 하는 이웃 샘플링뿐
- 평가 : 소수 클래스 F1
- 트리 모델 전처리 (보충자료)
  - 입력 피처
    - 기본 거래 피처 : 금액, 통화, 날짜, 시각, 결제 방식
    - 그래프 피처 : GFP(Graph Feature Preprocessor)로 생성
      - scatter-gather 패턴 (시간 창 6시간)
      - temporal cycle (길이 10 이하의 단순 사이클, 시간 창 1일)
      - 계좌(노드) 통계 : 금액·시각 기준 (시간 창 1일)
      - GFP는 실시간 스트리밍용 라이브러리라 메모리에 1일치만 들고 있어야 속도가 나오기 때문에 시간 창(6시간·1일)은 정확도가 아니라 GFP 실시간 처리량과의 타협으로 정한 값.
  - 누수 방지 : 거래를 시각 순서로 처리해서, 각 거래의 그래프 피처는 과거 거래로만 계산
    ("transactions from the training set will not contain graph-based features computed using the validation or test sets")
  - 배치 : 그래프 피처를 128건 단위로 추출
  - Large 다운샘플링 : 언급 없음
  - Large 분할 : 비율·방법 미기재 (권장 분할이 있다고만 적힘)
  - 튜닝 : successive halving으로 하이퍼파라미터 탐색. HI·LI-Large의 LightGBM은 좁은 범위(Range-small)로 탐색, 최적 설정으로 시드 4개 학습
  - 우리 실험과의 차이 : GFP 피처는 시간 창(6시간, 1일)을 두고 과거로만 계산. 우리 노드 피처는 시간 창 없이 누적(구간 끝까지).
    긴 구조(사이클·scatter-gather)를 피처로 미리 계산해 넣는 방식의 근거

### 2. Provably Powerful Graph Neural Networks for Directed Multigraphs (Egressy et al., AAAI 2024)
- Multi-GNN. v10 구조(역방향 메시지 패싱, Ego ID)의 원본
- 데이터셋 : HI·LI × Small, Medium ("two small and two medium-sized datasets")
- 분할 : 시간순 60/20/20. 학습 그래프는 학습 거래만, 검증 그래프는 학습+검증, 테스트 그래프는 전체
- 방식 :
  - 가중 손실 (minority class weight 튜닝. "has to be optimized to account for the greater class imbalance")
  - 다운샘플링·오버샘플링 없음
  - 이웃 샘플링 2홉 × 100
- 평가 : 소수 클래스 F1

### 3. FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection (Lin et al., ICAIF 2024)
- 코드 저장소: [junhongmit/FraudGT (GitHub)](https://github.com/junhongmit/FraudGT)
- 데이터셋 : HI·LI × Small, Medium, **Large** (HI-Large로 GNN을 학습한 직접 레퍼런스) 
- 분할 : 시간순 60/20/20 - 검증·테스트 그래프에 이전 거래 전부 포함
- 방식 :
  - **불균형 처리 — 가중 손실 (다운샘플링 없음)** 
    - `loss_fun: weighted_cross_entropy`, `loss_fun_weight`: [1, 6] (GINE·GT 계열 19개), [1, 7] (PNA 4개)
      - 손실 계산에서 정상 거래 틀림 = 1점, 세탁 거래 틀림 = 6점 (PNA 모델 4개만 7)
    - Small·Medium·Large 전부 같은 값. Altman 2023의 “6~8"과 일치
    - 학습 target은 전체 학습 간선에서 무작위 셔플(`LinkNeighborLoader`, `shuffle=True`). 양성 비율 조정 없음 → 양성이 자연 비율(0.124%)로 배치에 들어옴
      - 학습 간선 1.08억 건을 섞어서 앞에서부터 2048개씩 자른다 → 배치 안 세탁 비율은 데이터 그대로 0.124%, 즉 2048개 중 평균 2.5건
    - HI-Large에서 π_eff 환산 ≈ 6×0.00124 / (6×0.00124 + 0.99876) ≈ 0.74%
  - **이웃 샘플링**
    - `neighbor_sizes`: [50, 50] — 2홉 × 50 (Multi-GNN의 100이 아님)
      - 1홉: 최대 50개
      - 2홉: 1홉에서 뽑힌 50개 각각이 또 50개씩 → 최대 2,500개
      - 합계 최대 2,550개 = **target 간선은 양끝 계좌 둘이니 ×2 → 최대 5,100개**
    - `batch_size`: 2048 target 간선이 샘플링한 서브그래프를 공유 (v10처럼 target당 서브그래프를 따로 만들지 않음)
      - target 간선 2048개를 한 번에 잡고, 그 2048개 양끝 계좌(최대 4096개)에서 시작해 이웃을 한 번에 샘플링해 서브그래프 하나를 만든다. 겹치는 이웃은 한 번만 들어가고, 메시지 패싱도 한 번만 돌린 뒤 2048개 간선 점수를 한꺼번에 읽음. 서브그래프 수가 1.08억 → 5.3만(=1.08억/2048)으로 줄어듬
    - `add_ego_id`: True, 
      - 서브그래프 안에서 "지금 판정 대상 간선의 양끝 계좌"에 1, 나머지에 0인 피처 한 칸을 붙여서 모델이 "내가 지금 누굴 보고 있는지" 알게 하기
    `reverse_mp`: True, 
      - 거래는 A→B 방향이 있는데, 기본 메시지 패싱은 B가 A의 정보를 받기만 하고 A는 B를 못 보기 때문에 `역방향 간선(rev_to)`을 추가해서 "돈을 보낸 쪽도 받은 쪽 사정을 볼 수 있도록 하기
    `add_ports`: True (설정별)
      - 계좌별로 상대 계좌에 번호를 매겨 간선 피처로 붙여. 예를 들어 A가 B, C, B, D와 거래했으면 B=0, C=1, D=2 식으로. 같은 상대와 반복 거래하는지(multigraph에서 평행 간선 구분)를 모델이 알게 하기
  - **에폭 정의**
    - `iter_per_epoch`: 256 (14개는 128) × batch_size: 2048 = 에폭당 52만 건 (26만 건)
      - HI-Large 학습 간선은 약 1.08억 건이니 이 "에폭"은 전체의 **0.5%**만 본 것
    - `max_epoch`: 500 → 총 target 방문 ≈ 2.6억 건. HI-Large 학습 간선 ≈ 1.08억 건이니 전체 데이터를 약 2.4회 지나는 예산
    - `LoaderWrapper`가 셔플된 iterator를 에폭 경계에서 끊지 않고 이어감 → "에폭"은 전체 1회 통과를 256 배치씩 잘라 부르는 이름일 뿐
    - 즉 정상을 지우진 않지만 학습 예산을 고정해서 1회 통과조차 안 하는 구간이 대부분 — 우리 "지도 샘플링 r 축소 + 여러 에폭"과 총 양성 노출 수 관점에선 비슷한 효과
- HI-Large 테스트 F1 : GIN 36.3 · GIN+RMP 59.3 · GIN+Ego 53.3 · Multi-GIN 58.4 · LightGBM+그래프피처 58.0 · Multi-FraudGT 73.3

### 4. BlazingAML: High-Throughput Anti-Money Laundering via Multi-Stage Graph Mining (2026, arXiv 2604.12241)
- 데이터셋 : HI·LI × Small, Large (Medium 사용 여부 미확인)
- 분할 : 시간순 80/20
- 모델 : XGBoost + 그래프 마이닝 피처 (Fan, Degree, Cycle, Scatter-Gather). GNN 아님
- 방식 : 다운샘플링·업샘플링·가중치 모두 미기재. 불균형은 평가 지표(F1)로만 언급
- HI-Large F1 : XGB 단독 20.2 → 그래프 피처 전부 58.1, FraudGT 54.3 (비교 기준)

### 5. Graph Neural Network with One-side Edge Sampling for Fraud Detection (2026, arXiv 2601.06800)
- 데이터셋 : HI·LI × Small
- 방식 : One-side Edge Sampling(OES)
  - 학습 중 모델이 이미 높은 확신으로 맞히는 간선을 학습 그래프에서 제거 (구조 샘플링 계열)
  - 검증·테스트는 전체 그래프
- 결과 : HI-Small GIN+EU F1 50.96 → 59.47. 다른 백본은 개선이 미미 (표준편차 범위 안)
- 참고 : 논문 안에서 "샘플링한다/제거한다" 서술이 섞여 있고, 개선 폭이 일관되지 않음

### 6. Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection (Blanuša et al., 2024, arXiv 2402.08593)
- IBM GFP 논문. Altman 2023 보충자료의 트리 모델 전처리를 별도 논문으로 낸 것
- 데이터셋 : AML HI·LI × Small, Medium, Large + ETH Phishing. 결과표에 "AML Large" 열이 있음 (HI/LI 구분은 표에서 미확인)
- 분할 : 시간순 60/20/20
- 방식 :
  - 트리 모델(LightGBM, XGBoost)의 scale_pos_weight 탐색 범위 1~10. 선택값 미기재
  - 다운샘플링·오버샘플링 없음
- 참고 : Large 규모에서도 정상을 줄이지 않고 scale_pos_weight만으로 처리. 단 트리 모델이라 GNN처럼 1에폭 비용 문제가 없음

### 7. MEGA: Message Passing Neural Networks for Multigraphs with EdGe Attributes (2024, arXiv 2412.00241)
- IBM 쪽 Multi-GNN 후속
- 데이터셋 : HI·LI × Small, Medium (Large 미사용)
- 분할 : "same temporal split as the Multi-GNN baseline"
- 방식 :
  - 하이퍼파라미터 표에 클래스 가중치 w (AML 1, 6.27 / ETH 1, 7). 본문에 가중 손실이라는 서술은 없고 표 값으로만 확인
  - 다운샘플링·오버샘플링 없음
  - 이웃 샘플링 2홉 × 100 (Multi-GNN과 동일)

### 8. Privacy-Preserving Graph-Based Machine Learning with FHE for Collaborative AML (2024, arXiv 2411.02926)
- 데이터셋 : HI-Small을 두 개의 소형 subset으로 축소
  - Dataset 1 : 거래 10,354건, 세탁 50% — "Undersampling was used to create a balanced dataset"
  - Dataset 2 : 거래 5,491건, 세탁 5.72% — 패턴 그룹 370개 중 40개 무작위 선택
- 모델 : GFP 그래프 피처 + Bayesian-optimized XGBoost. GIN은 FHE 연산자 미지원으로 미완
- 결과 (Dataset 2, F1) : 기본 피처 0.31 → +1홉 피처 0.39 → +다홉 0.37 → +노드 통계 0.35
- 참고 : IBM AML 데이터에서 정상을 실제로 줄인 유일한 사례이지만, 목적이 **FHE 추론 비용 절감**이지 불균형 해소가 아님.
  축소 후 거래가 1만 건 이하라 우리 규모(8,750만)와 비교 대상이 아님

## 정리
| 논문 | HI-Large 사용 | 불균형 처리 | 그래프에서 거래 제거 |
|---|---|---|---|
| Altman 2023 | 트리 모델만 (보충자료) | 가중 손실 / scale_pos_weight | 없음 |
| Egressy 2024 | 없음 | 가중 손실 | 없음 |
| FraudGT 2024 | GNN | BCE (가중치 미기재) | 없음 |
| BlazingAML 2026 | XGBoost | 미기재 | 없음 |
| OES 2026 | 없음 | 쉬운 간선 제거 | 있음 (학습 그래프만) |
| GFP 2024 | 트리 모델 (Large 열 존재) | scale_pos_weight 1~10 | 없음 |
| MEGA 2024 | 없음 | 가중 손실 (w=6.27, 표에서만 확인) | 없음 |
| FHE-AML 2024 | 없음 (HI-Small 1만 건으로 축소) | 언더샘플링 (균형 subset) | 있음 (전체 데이터 축소) |

- 대부분이 **그래프를 유지하고 손실 가중치로** 불균형을 다룸. 다운샘플링을 쓴 연구는 없음
- HI-Large로 GNN을 학습한 건 FraudGT뿐이고, 정상 거래를 줄이지 않고 이웃 샘플링만 사용
- 구조 샘플링 계열은 OES 하나이고, 쉬운 간선을 지우는 방식이라 우리 구조 샘플링(정상 무작위 제거)과는 목적이 다름
- IBM AML 데이터에서 정상을 실제로 줄인 논문은 FHE-AML 하나이고, 이유는 불균형이 아니라 암호화 연산 비용. 불균형 해소 목적의 다운샘플링 선례는 여전히 없음
- IBM 계열(Altman·Egressy·GFP·MEGA)은 일관되게 "그래프 유지 + 손실 가중치(GNN 6~8, 트리 1~10)" 방식
- 미확인 후보 : "Synthetic Graph Data Generation to Mitigate Class Imbalance in Money Laundering Detection" (ResearchGate 399720414, 본문·데이터셋 미확인), GraphSAGE+ (Ghahremani & Nguyen, ICCPR 2025, 패턴 기반 오버샘플링, 유료라 미확인)

## 출처
- [Realistic Synthetic Financial Transactions for Anti-Money Laundering Models (arXiv)](https://arxiv.org/html/2306.16424v1) - [NeurIPS 2023 Supplemental](https://papers.neurips.cc/paper_files/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Supplemental-Datasets_and_Benchmarks.pdf)
- [Provably Powerful Graph Neural Networks for Directed Multigraphs (arXiv)](https://arxiv.org/html/2306.11586v3) · [IBM/Multi-GNN 코드](https://github.com/IBM/Multi-GNN)
- [FraudGT (PDF)](https://jshun.csail.mit.edu/FraudGT.pdf)
- [BlazingAML (arXiv 2604.12241)](https://arxiv.org/pdf/2604.12241)
- [Graph Neural Network with One-side Edge Sampling for Fraud Detection (arXiv 2601.06800)](https://arxiv.org/pdf/2601.06800)
- [Graph Feature Preprocessor (arXiv 2402.08593)](https://arxiv.org/html/2402.08593v2)
- [MEGA (arXiv 2412.00241)](https://arxiv.org/pdf/2412.00241)
- [Privacy-Preserving Graph-Based ML with FHE for Collaborative AML (arXiv 2411.02926)](https://arxiv.org/html/2411.02926v2)
