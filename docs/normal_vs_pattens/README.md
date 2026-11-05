# 정상 vs 무패턴 비교

`code/nonpaten_vs_normal` 노트북별 분석 정리. 모두 분포 비교이며 분류 성능은 아직 확인하지 않았다.

| 문서 | 노트북 | 내용 |
|---|---|---|
| [02](02_pair_history_amount_receiver_analysis.md) | `02_pair_history_amount_receiver_analysis.ipynb` | 과거 쌍 이력 유무, 월별 변화, 금액 z, 수신 계좌 활동량 |
| [03](03_pair_bins_same_minute_analysis.md) | `03_pair_bins_same_minute_analysis.ipynb` | 과거 거래 5개 구간, 같은 분 활동, 월별 구간 비율 변화 |
| [04](04_upstream_same_minute_diagnosis.md) | `04_upstream_same_minute_diagnosis.ipynb` | 집중 송신 계좌, 상류 출처, 같은 분 같은 쌍 쌍둥이 |

공통 기준: 정상은 `is_laundering = False`, 무패턴은 라벨 1이면서 `patterns.parquet`에 연결되지 않은 거래. 02·03은 패턴 양성을 비교에서 빼고, 04는 패턴 양성도 집계에 넣었다(이진 모델 운영점 기준).

## 정리와 다음 단계

| 후보 | 결과 | 결정 |
|---|---|---|
| 누적 과거 쌍 건수 | 1~5건 구간 비율이 월별로 7~22배 변함 | 고정 창(최근 7일) 건수를 함께 둔다 |
| 같은 분 같은 쌍 | 새 쌍 칸의 양성 비율 50%, 양성 쪽 금액이 큰 경우 70% | **피처로 만든다** — 같은 분 다른 거래 수, 금액 로그비, 단독 최대 여부 |
| 같은 분 수신 계좌 출금 | 출금이 있으면 무패턴이 거의 없음 | 보조 피처로 둔다 |
| 상류 출처 | 배수 1.5~2.0, hot은 1.0 | 접는다 |
| hot 15개 계좌 | 모든 축에서 무패턴 비율 0.14%로 평평 | 평가 때 따로 떼서 본다 (무패턴 약 28%) |

다음 [05번 노트북](../../code/nonpaten_vs_normal/05_nonpattern_features.ipynb)에서 팀원 신규 8종(12열)과 같은 분 같은 쌍 6열을 만들고, 06에서 LightGBM으로 기존 v10 피처와 비교한다. 위 수치는 모두 분포 비교이며 분류 성능은 아직 확인하지 않았다.
