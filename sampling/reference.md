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

### 4. BlazingAML: High-Throughput Anti-Money Laundering via Multi-Stage Graph Mining (Ye et al., 2026, arXiv 2604.12241)
- 데이터셋 : HI·LI × Small, Medium, Large 6개 전부
- 분할 : 시간순 80/20 (검증 분할 없음)
- 모델 : XGBoost + 그래프 마이닝 피처 (Fan-in/out, Degree, Cycle, Scatter-Gather 참여 횟수). GNN 아님. 시간 창 사용하나 값 미기재
- 방식 : 다운샘플링·업샘플링·scale_pos_weight 모두 미기재. 불균형은 F1 채택 이유로만 언급
- HI-Large F1 (Table 2 피처 누적) : XGB 단독(계좌 ID 피처) 20.2 → +Fan 39.1 → +Degree 55.7 → +Cycle 57.5 → +SG 58.1
- FraudGT 비교 (Table 4, LI-S/HI-S/LI-M/HI-M/LI-L/HI-L) : FraudGT 28.6/69.6/24.0/62.3/11.0/54.3 vs BlazingAML 18.4/46.4/21.4/51.1/17.8/58.1
  - Large 두 개에서만 BlazingAML이 앞섬. Small·Medium은 FraudGT가 10~20p 높음
  - FraudGT HI-Large 54.3은 FraudGT 논문 자체 수치(Multi-FraudGT 73.3)와 다름. 재실행인지 인용인지 미기재 → 직접 비교 불가

### 5. Graph Neural Network with One-side Edge Sampling for Fraud Detection (Trieu, 2025~, arXiv 2601.06800)
- 단독 저자·소속·날짜 미기재. 근거로 쓰기엔 신뢰도 낮음 (아래 불일치 참고)
- 데이터셋 : HI·LI × Small. 분할·그래프 구성은 Multi-GNN과 동일 (검증·테스트는 이전 거래 포함, OES는 학습에만)
- 방식 : One-side Edge Sampling(OES) — 구조 샘플링 계열
  - 매 에폭, 확신도(max 클래스 확률) 상위 p=99 퍼센타일이면서 정답인 간선 E′를 뽑아 비율 r=10으로 샘플링 → 이 그래프가 학습 그래프를 **통째로 대체** (메시지 패싱 그래프 자체가 줄어듦)
  - 에폭 n=20까지만 적용
  - 불일치 : 동기는 "쉬운 간선 제거"인데 알고리즘은 쉬운 간선을 남기는 것으로 읽힘. 제거 비율 공식 (1−p)·r 도 값이 안 맞음
- 불균형 처리 : 없음 (가중 손실·비율 조정 모두 없음)
- 결과 (5시드 평균 F1) : HI-Small GIN+EU 50.96±6.95 → 59.47±4.47 만 유의미. GIN 33.4→37.4, GIN+Ego 35.3→35.4, LI-Small 3개 모두 표준편차 안
- 학습 시간 : 8~18% 단축 (RTX A4000 1장)
- 우리와의 관계 : "학습 그래프에서 간선을 지워도 일부 백본은 성능 유지"라는 약한 데이터 포인트. 정상 무작위 제거인 우리 구조 샘플링과는 선택 기준이 다름

### 6. Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection (Blanuša et al., ICAIF 2024, arXiv 2402.08593)
- IBM GFP 논문. Altman 2023 보충자료의 트리 모델 전처리를 별도 논문으로 낸 것
- 그래프를 모델에 넣지 않고, 거래 스트림을 흘려보내며 거래마다 그래프 피처를 계산해 표로 만든 뒤 트리 모델에 넣는 방식
- 데이터셋 : AML HI/LI x Small, Medium, Large 6개 전부 (HI-Large 2.1M 계좌 / 180M 거래 / 세탁 0.124% / 97일) + ETH Phishing (실제 이더리움 피싱 데이터, 우리와 무관)
- 방식
  - 스트리밍 : 거래를 시각순 배치(128 또는 2048건)로 넣고, 현재 시각 기준 과거 일정 시간 안의 간선만 메모리에 유지. AML은 scatter-gather 6시간, 나머지 1일 (6시간만 짧은 이유는 미기재, "처리량과 정확도의 트레이드오프"라고만)
  - 누수 방지 : 학습 구간 거래만으로 학습 피처를 계산한 뒤 검증 구간을 이어서 처리 ("computed on the graph created using only those transactions")
  - 라벨은 데이터에 있는 Is Laundering 0/1 하나. 이진 분류. 패턴 유형 라벨은 안 씀
  - 분할 : 시간순 60/20/20. LightGBM 3.1.1 / XGBoost 1.7.5, successive halving으로 튜닝 (Large는 초기 조합 16개), early stopping 20, 검증으로 선택, 테스트 1회, 5회 평균
  - 불균형 : scale_pos_weight 탐색 범위 1~10. 선택값과 임계값 미기재. 다운샘플링, 오버샘플링 없음
- 피처가 들어가는 형태 : 거래 한 건 = 표의 한 줄이고, 그 줄 오른쪽에 숫자 열이 4묶음 붙는다. 전부 라벨을 안 보고 정상 거래까지 포함해서 센 값이다 (라벨을 보고 세면 추론 때 같은 피처를 만들 수 없고 누수가 된다). 열 이름은 논문에 없고 아래는 설명용으로 지은 이름
  - 예시 거래 : A -> E, 5,000 USD
  1. fan-in / fan-out : 계좌 단위로 세서 거래 행에 복사. A가 (1일 안에) 몇 곳에 보냈나 / 몇 곳에서 받았나, E도 마찬가지. 숫자 2개씩
     `A_fan_out=3  A_fan_in=1  E_fan_out=0  E_fan_in=4`
  2. scatter-gather : 이 거래가 "A -> 여러 곳 -> E" 모양에 속하는지. 중간 계좌 수별로 칸을 나눠 셈 (bin)
     `sg_중간2개=1  sg_중간3개=0  sg_중간4개이상=0`
     - gather-scatter(여러 곳 -> A -> 여러 곳)는 별도 피처 없음. A의 fan-in, fan-out이 둘 다 2 이상이면 그걸로 잡힌다고 봄
  3. 사이클 : 이 거래가 길이 2~10짜리 고리에 몇 번 속하는지, 길이별 bin. 단순 사이클(순서 무관)과 시간순 사이클(돈이 순서대로 돈) 두 종류
     `cycle_len2=0  cycle_len3=1  cycle_len4=0 ...  temporal_cycle_len3=1 ...`
  4. 계좌 통계 : 출발 계좌 A와 도착 계좌 E 각각. 1일 안에 그 계좌가 받은/보낸 금액의 합, 평균, 분산, 왜도, 첨도. 시각에 대해서도 같은 통계. A 통계와 E 통계가 같은 행에 나란히 들어감
     `A_sent_sum=12000  A_sent_mean=4000  A_sent_var=...  E_received_sum=20000 ...`
  - 계좌 ID 제외 : A, E가 누구인지(계좌 번호)는 피처에서 뺌. BlazingAML은 반대로 ID만 넣은 걸 기준선으로 삼음
  - 1, 4번은 계좌(노드) 단위로 계산해 거래 행에 복사한 것이고 2, 3번은 거래(엣지) 단위 계산. 단 1, 4번도 "그 거래 시각 기준 1일 창"이라 같은 계좌라도 거래마다 값이 달라짐 (우리 노드 피처는 구간 누적값 하나)
  - "Degree" 피처는 GFP에 없음 (BlazingAML 표의 Degree 열이 뭘 가리키는지 불명)
- 결과 (소수 클래스 F1, Table 4)
  - HI-Large : LightGBM basic 24.5 -> GFP+LightGBM 58.0 (batch 128) / XGBoost basic 10.6 -> GFP+XGBoost 56.9 (batch 2048). GNN은 Large 결과 NA
  - HI-Small : GFP+XGBoost 64.8 vs PNA 56.8 / GIN+EU 47.7 (GNN 수치는 Altman 2023 인용)
- 피처 ablation (Table 5, HI-Small XGB) : basic 19.8 -> +fan 59.7 -> +사이클/SG 61.0 -> +계좌 통계 64.8. 본문 표현은 fan "more than 30%", 다중홉 "up to 4%"
- SHAP 상위 2개 (HI-Small) : 2홉 시간순 사이클 수, 도착 계좌의 받은 금액 합
- 참고 : 시간 창(6시간, 1일)은 처리량과의 타협으로 정한 값. 창 크기가 탐지에 미치는 영향, 세탁 패턴 중 창 안에 드는 비율은 논문에 없음 -> 우리가 모티프 피처를 넣을 땐 패턴 시간 폭을 재서 창을 정해야 함
- 우리와의 관계
  - fan-in/out이 효과의 대부분이고 사이클/SG는 +1~4p -> 모티프 피처 우선순위 낮음 (BlazingAML의 +2.4p와 일치)
  - 계좌 통계 +4~8p는 우리 노드 피처(mean_paid_currency_z, paid_outlier_fraction 등)와 같은 계열. 단 GFP는 1일 창, 우리는 구간 누적
  - 학습 구간 피처 분리는 우리 features_train/val/test와 같은 논리

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
