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


### 현재 피처에서 전구간 누적을 해서 사용하는 피처는

#### 노드 피처 19개
- 거래 수·상대 수 : log_out_count, log_in_count, log_out_neighbors, log_in_neighbors, log_self_count, log_reciprocal_neighbors
- 금액 : mean_paid_currency_z, paid_outlier_fraction
- 기타 : log_currency_count, log_active_days, log_entity_observed_accounts, log_entity_out_diversity, log_entity_in_diversity
- 누적 아님 : type_* 6개 (Corporation, Partnership, Sole Proprietorship, Individual, Country, Direct)

### 엣지 피처 4개
- pair_prior_count : 같은 쌍, 그 시각까지
- pair_history_count : 같은 쌍, 전날까지
- reverse_history_count : 반대 방향, 전날까지
- pair_age_at_day_start : 같은 쌍 첫 거래부터 지난 시간

엣지 중 창·고정값 (누적 아님): pair_gap_minutes, pair_count_1h, sender_gap_minutes, sender_count_1h, receiver_gap_minutes, receiver_count_1h, paid_currency_z, amt_z_pay_ccy, same_entity, same_known_country, both_countries_known, same_currency, is_self_transfer