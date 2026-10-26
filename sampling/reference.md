## 선행 연구의 클래스 불균형 처리

### 1. Realistic Synthetic Financial Transactions for Anti-Money Laundering Models (Altman et al., NeurIPS 2023)
- AMLworld 데이터셋 원 논문
- 데이터셋 : 본문은 HI·LI × Small, Medium (GNN·트리 모델 모두)
  보충자료에서 HI·LI × Large를 트리 모델(GFP+XGBoost, LightGBM)로만 추가 실험. GNN의 Large 결과는 없음
- 분할 : 시간순 60/20/20. 검증·테스트 그래프에 이전 거래 포함
- 방식 :
  - GNN - 가중 손실 (minority class weight 튜닝, 탐색 범위 6~8)
  - 트리 모델 - scale_pos_weight (탐색 범위 1~10)
  - 다운샘플링·오버샘플링 없음. 샘플링은 규모 때문에 하는 이웃 샘플링뿐
- 평가 : 소수 클래스 F1
- 트리 모델 전처리 (보충자료)
  - 입력 피처
    - 기본 거래 피처 : 금액, 통화, 날짜, 시각, 결제 방식
    - 그래프 피처 : GFP(Graph Feature Preprocessor)로 생성
      - scatter-gather 패턴 (시간 창 6시간)
      - temporal cycle (길이 10 이하의 단순 사이클, 시간 창 1일)
      - 계좌(노드) 통계 : 금액·시각 기준 (시간 창 1일)
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
- 데이터셋 : HI·LI × Small, Medium, **Large** (HI-Large로 GNN을 학습한 직접 레퍼런스)
- 분할 : 시간순 60/20/20
- 방식 :
  - 손실은 binary cross-entropy로만 기술. 클래스 가중치·다운샘플링 사용 여부는 미기재
  - 이웃 샘플링 사용 (홉 수·개수 미기재)
  - 불균형은 평가 지표(F1)로만 언급
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

## 정리
| 논문 | HI-Large 사용 | 불균형 처리 | 그래프에서 거래 제거 |
|---|---|---|---|
| Altman 2023 | 트리 모델만 (보충자료) | 가중 손실 / scale_pos_weight | 없음 |
| Egressy 2024 | 없음 | 가중 손실 | 없음 |
| FraudGT 2024 | GNN | BCE (가중치 미기재) | 없음 |
| BlazingAML 2026 | XGBoost | 미기재 | 없음 |
| OES 2026 | 없음 | 쉬운 간선 제거 | 있음 (학습 그래프만) |

- 대부분이 **그래프를 유지하고 손실 가중치로** 불균형을 다룸. 다운샘플링을 쓴 연구는 없음
- HI-Large로 GNN을 학습한 건 FraudGT뿐이고, 정상 거래를 줄이지 않고 이웃 샘플링만 사용
- 구조 샘플링 계열은 OES 하나이고, 쉬운 간선을 지우는 방식이라 우리 구조 샘플링(정상 무작위 제거)과는 목적이 다름

## 출처
- [Realistic Synthetic Financial Transactions for Anti-Money Laundering Models (arXiv)](https://arxiv.org/html/2306.16424v1) - [NeurIPS 2023 Supplemental](https://papers.neurips.cc/paper_files/paper/2023/file/5f38404edff6f3f642d6fa5892479c42-Supplemental-Datasets_and_Benchmarks.pdf)
- [Provably Powerful Graph Neural Networks for Directed Multigraphs (arXiv)](https://arxiv.org/html/2306.11586v3) · [IBM/Multi-GNN 코드](https://github.com/IBM/Multi-GNN)
- [FraudGT (PDF)](https://jshun.csail.mit.edu/FraudGT.pdf)
- [BlazingAML (arXiv 2604.12241)](https://arxiv.org/pdf/2604.12241)
- [Graph Neural Network with One-side Edge Sampling for Fraud Detection (arXiv 2601.06800)](https://arxiv.org/pdf/2601.06800)
