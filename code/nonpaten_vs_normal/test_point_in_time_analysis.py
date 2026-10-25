"""실제 데이터 없이 시간 누수·패턴 이력 포함·집계/그림을 검증한다.

실행: python -m unittest discover -s code/nonpaten_vs_normal -p 'test_*.py' -v
"""
import contextlib
from datetime import datetime, timedelta
import io
import math
from pathlib import Path
import tempfile
import unittest

import duckdb
import pandas as pd
import point_in_time_analysis as pit


class PointInTimeTest(unittest.TestCase):
    def setUp(self):
        # 작업 폴더 안에서만 생성·정리한다. 실제 입력 데이터는 사용하지 않는다.
        self.tmp = tempfile.TemporaryDirectory(prefix="aml-pit-test-", dir=str(Path.cwd()))
        self.root = Path(self.tmp.name).resolve()
        self.assertEqual(self.root.parent, Path.cwd().resolve())
        self.addCleanup(self.tmp.cleanup)
        self.con = duckdb.connect()
        self.addCleanup(self.con.close)
        self.rows = [
            (1,"2022-08-01 09:00",1,2,10.,"USD",False),
            (2,"2022-08-01 09:00",3,2,100.,"USD",True),  # 패턴: 보정·이력에 포함
            (3,"2022-08-01 10:00",2,2,30.,"USD",False),  # 자기 이체
            (4,"2022-08-01 11:00",2,4,40.,"EUR",False),
            (5,"2022-08-02 10:00",1,2,50.,"USD",True),   # 패턴: 이력에 포함
            (6,"2022-08-02 10:00",7,2,60.,"USD",False),
            (7,"2022-08-02 10:30",1,2,70.,"USD",True),
            (8,"2022-08-02 11:00",1,2,80.,"USD",False),
            (9,"2022-08-02 11:00",9,2,90.,"USD",True),
            (10,"2022-08-02 11:01",2,2,100.,"USD",True),
            (11,"2022-08-02 12:01",10,2,110.,"USD",False),
            (12,"2022-08-02 12:02",7,99,20.,"YEN",True), # 기준 통화·계좌 이력 없음
            (13,"2022-08-03 00:00",1,2,120.,"USD",False),
            (14,"2022-08-03 00:01",5,2,130.,"USD",True),
            (15,"2022-08-03 01:00",6,4,55.,"EUR",True), # 기준 IQR=0, std=NULL
            (16,"2022-08-02 11:00",9,2,95.,"USD",True), # 같은 분의 같은 계좌쌍도 서로 제외
        ]

    def execute_pipeline(self, rows, tag):
        folder = self.root / tag
        folder.mkdir()
        con = self.con
        con.execute("""CREATE OR REPLACE TABLE input_rows (
          edge_id BIGINT,ts TIMESTAMP,src BIGINT,dst BIGINT,paid DOUBLE,
          payment_currency VARCHAR,is_laundering BOOLEAN)""")
        con.executemany("INSERT INTO input_rows VALUES (?,?,?,?,?,?,?)",rows)
        tx, pat = folder/'transactions.parquet', folder/'patterns.parquet'
        con.execute(f"COPY (SELECT *,'ACH' AS payment_format FROM input_rows) TO {pit.q(tx)} (FORMAT PARQUET)")
        con.execute(f"COPY (SELECT edge_id,ts,FALSE AS unmatched FROM input_rows WHERE edge_id IN (2,5)) TO {pit.q(pat)} (FORMAT PARQUET)")
        with contextlib.redirect_stdout(io.StringIO()):
            audit = pit.prepare_inputs(con,tx,pat,'2022-08-01','2022-08-02','2022-08-02','2022-08-04')
            pit.fit_amount_reference(con,'2022-08-01','2022-08-02')
            pit.build_history_states(con)
            audit.update(pit.write_features(con,folder/'features.parquet',audit['target_rows']))
        return con.execute('SELECT * FROM features ORDER BY edge_id').df(), folder, audit

    def test_features_against_direct_history_and_reports(self):
        df, folder, audit = self.execute_pipeline(self.rows,'base')
        indexed = df.set_index('edge_id')
        # 각 거래의 과거 행을 직접 나열하는 별도 기준으로 SQL 누적 집계를 검증한다.
        rows = [dict(zip(['edge_id','ts','src','dst','paid','currency','label'],r)) for r in self.rows]
        for row in rows:
            row['ts'] = datetime.fromisoformat(row['ts'])
        for row in rows:
            if row['ts'] < datetime(2022,8,2) or row['edge_id'] in (2,5):
                continue
            past = [r for r in rows if r['ts'] < row['ts']]
            incoming = [r for r in past if r['dst']==row['dst']]
            outgoing = [r for r in past if r['src']==row['dst']]
            pair = [r for r in past if (r['src'],r['dst'])==(row['src'],row['dst'])]
            days = {r['ts'].date() for r in incoming+outgoing}
            observed = indexed.loc[row['edge_id']]
            expected = {
                'pair_prior_n':len(pair), 'receiver_in_count':len(incoming),
                'receiver_out_count':len(outgoing),
                'receiver_in_neighbors':len({r['src'] for r in incoming}),
                'receiver_out_neighbors':len({r['dst'] for r in outgoing}),
                'receiver_active_days':len(days),
                'receiver_in_count_1h':sum(r['ts']>=row['ts']-timedelta(hours=1) for r in incoming),
                'receiver_out_count_1h':sum(r['ts']>=row['ts']-timedelta(hours=1) for r in outgoing),
            }
            for name,value in expected.items():
                self.assertEqual(observed[name],value,(row['edge_id'],name))
            self.assertAlmostEqual(observed.pair_prior_count,math.log1p(len(pair)))
            self.assertEqual(observed.pair_group,'one_or_more' if pair else 'zero')
            if incoming:
                gap = (row['ts']-max(r['ts'] for r in incoming)).total_seconds()/60
                self.assertEqual(observed.receiver_in_gap_minutes,gap)
            else:
                self.assertTrue(pd.isna(observed.receiver_in_gap_minutes))
            if days:
                self.assertAlmostEqual(observed.receiver_in_per_active_day,len(incoming)/len(days))
            else:
                self.assertTrue(math.isnan(observed.receiver_in_per_active_day))
        self.assertEqual(indexed.loc[7,'pair_prior_n'],2)  # 패턴 id5도 포함
        self.assertEqual(indexed.loc[6,'receiver_in_count'],3)  # 같은 분 id5 제외
        self.assertEqual(indexed.loc[8,'receiver_in_count_1h'],3) # 60분 전 2건 포함
        self.assertEqual(indexed.loc[11,'receiver_in_count_1h'],1) # 경계의 자기 이체 포함
        self.assertEqual(indexed.loc[13,'receiver_active_days'],2) # 현재 날짜 미포함
        self.assertEqual(indexed.loc[14,'receiver_active_days'],3)
        self.assertTrue(math.isnan(indexed.loc[12,'paid_currency_z']))
        self.assertTrue(math.isnan(indexed.loc[15,'amt_z_pay_ccy']))
        self.assertEqual(self.con.execute("SELECT scale FROM amount_reference WHERE payment_currency='USD'").fetchone()[0],30.)
        self.assertEqual(audit['normal_rows'],4)
        self.assertEqual(audit['nonpattern_rows'],7)
        self.assertEqual(audit['robust_z_missing'],1)
        self.assertEqual(indexed.loc[9,'pair_prior_n'],0)
        self.assertEqual(indexed.loc[16,'pair_prior_n'],0)
        import numpy as np
        from statistics import stdev, mean
        z_reference = np.log1p(np.array([10.,100.,30.])/30.)
        center = np.median(z_reference)
        iqr = np.quantile(z_reference,.75)-np.quantile(z_reference,.25)
        z_expected = np.clip((math.log1p(70./30.)-center)/iqr,-8,8)
        self.assertAlmostEqual(indexed.loc[7,'paid_currency_z'],z_expected)
        log_reference = [math.log1p(v) for v in [10.,100.,30.]]
        z_expected = (math.log1p(70.)-mean(log_reference))/(stdev(log_reference)+1e-6)
        self.assertAlmostEqual(indexed.loc[7,'amt_z_pay_ccy'],z_expected)
        with contextlib.redirect_stdout(io.StringIO()): tables = pit.summarize(self.con,folder)
        self.assertEqual(tables['group_counts'].n.sum(),len(df))
        self.assertEqual(tables['monthly_counts'].n.sum(),len(df))
        self.assertEqual(tables['currency_format_counts'].n.sum(),len(df))
        self.assertEqual(tables['z_histogram'].n.sum(),len(df))
        for _,part in tables['group_counts'].groupby('class_name'):
            self.assertAlmostEqual(part.share_within_class.sum(),1.)
        # 집계 결과만 사용하는 그림도 실제로 생성해 확인한다.
        import matplotlib
        matplotlib.use('Agg')
        pit.plot_results(tables,folder)
        self.assertEqual(len(list(folder.glob('*.png'))),3)

    def test_future_invariance_and_labels_do_not_change_history(self):
        import pandas as pd
        before,_,_ = self.execute_pipeline(self.rows,'before')
        future = (100,'2022-08-03 20:00',1,2,1e12,'USD',False)
        after,_,_ = self.execute_pipeline(self.rows+[future],'after')
        pd.testing.assert_frame_equal(before,after[after.edge_id!=100].reset_index(drop=True))
        changed = [(*r[:6],not r[6]) if r[0] not in (2,5) else r for r in self.rows]
        flipped,_,_ = self.execute_pipeline(changed,'flipped')
        cols=['edge_id','pair_prior_count']+pit.METRICS
        pd.testing.assert_frame_equal(before[cols],flipped[cols])

    def test_reference_period_cannot_overlap_analysis(self):
        with self.assertRaisesRegex(ValueError,'기준 기간'):
            pit.prepare_inputs(self.con,'unused','unused','2022-08-01','2022-08-09','2022-08-08')


if __name__ == '__main__':
    unittest.main()
