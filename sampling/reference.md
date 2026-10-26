## 선행 연구의 클래스 불균형 처리

### 1. Realistic Synthetic Financial Transactions for Anti-Money Laundering Models (Altman et al., NeurIPS 2023)
- AMLworld 데이터셋 원 논문
- 데이터셋 : HI·LI × Small, Medium (GNN) / HI·LI × Large는 트리 모델(LightGBM, XGBoost)만 (보충자료)
- 분할 : 시간순 60/20/20. 검증·테스트 그래프에 이전 거래 포함
- 방식 :
  - GNN — 가중 손실 (minority class weight 튜닝, 탐색 범위 6~8)
  - 트리 모델 — scale_pos_weight (탐색 범위 1~10)
  - 다운샘플링·오버샘플링 없음. 샘플링은 규모 때문에 하는 이웃 샘플링뿐
- 평가 : 소수 클래스 F1

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
| Altman 2023 | 트리 모델만 | 가중 손실 / scale_pos_weight | 없음 |
| Egressy 2024 | 없음 | 가중 손실 | 없음 |
| FraudGT 2024 | GNN | BCE (가중치 미기재) | 없음 |
| BlazingAML 2026 | XGBoost | 미기재 | 없음 |
| OES 2026 | 없음 | 쉬운 간선 제거 | 있음 (학습 그래프만) |

- 대부분이 **그래프를 유지하고 손실 가중치로** 불균형을 다룸. 다운샘플링을 쓴 연구는 없음
- HI-Large로 GNN을 학습한 건 FraudGT뿐이고, 정상 거래를 줄이지 않고 이웃 샘플링만 사용
- 구조 샘플링 계열은 OES 하나이고, 쉬운 간선을 지우는 방식이라 우리 구조 샘플링(정상 무작위 제거)과는 목적이 다름