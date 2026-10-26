"""Synthetic tests for 03: boundaries, self exclusion, labels, future invariance."""
from datetime import datetime, timedelta
from pathlib import Path
import contextlib
import io
import json
import math
import tempfile
import unittest

import duckdb
import pandas as pd
import same_minute_analysis as analysis


class SameMinuteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='aml-minute-test-', dir=Path.cwd())
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.con = duckdb.connect()
        self.addCleanup(self.con.close)
        t = datetime(2022,8,8,10)
        self.rows = []
        self.target_ids = []
        for j,(src,dst,prior) in enumerate([(1,2,0),(3,4,1),(5,6,2),(7,8,5),
                                          (9,10,6),(11,12,20),(13,14,21),(2,2,0),(15,2,0)]):
            for k in range(prior):
                self.rows.append((len(self.rows)+1,datetime(2022,8,1)+timedelta(minutes=k),
                                  src,dst,10.,'USD','ACH',k%2==0))
            edge_id = 1000+j
            self.rows.append((edge_id,t,src,dst,100.+j,'USD','ACH',j%2==1))
            self.target_ids.append(edge_id)
        # Known-pattern positives absent from targets but included in minute context.
        self.rows += [(2000,t,1,2,55.,'USD','ACH',True),
                      (2001,t,99,2,66.,'USD','ACH',True),
                      (2002,t,2,77,77.,'USD','ACH',True)]

    def fixture(self, tag, rows=None):
        rows = self.rows if rows is None else rows
        folder = self.root/tag
        folder.mkdir()
        con = self.con
        con.execute('''CREATE OR REPLACE TABLE raw_input (
            edge_id BIGINT,ts TIMESTAMP,src BIGINT,dst BIGINT,paid DOUBLE,
            payment_currency VARCHAR,payment_format VARCHAR,is_laundering BOOLEAN)''')
        con.executemany('INSERT INTO raw_input VALUES (?,?,?,?,?,?,?,?)',rows)
        tx = folder/'transactions.parquet'
        con.execute(f'COPY raw_input TO {analysis.q(tx)} (FORMAT PARQUET)')
        # Independent row-based implementation of strict prior history.
        targets = []
        for r in rows:
            if r[0] not in self.target_ids:
                continue
            past = [x for x in rows if x[1]<r[1]]
            pair = [x for x in past if (x[2],x[3])==(r[2],r[3])]
            incoming = [x for x in past if x[3]==r[3]]
            gap = (r[1]-max(x[1] for x in incoming)).total_seconds()/60 if incoming else None
            targets.append((*r,'nonpattern' if r[-1] else 'normal',len(pair),
                            .25 if r[0]!=1000 else None,.5,gap,len(incoming),math.log1p(len(pair))))
        con.execute('''CREATE OR REPLACE TABLE base_input (
            edge_id BIGINT,ts TIMESTAMP,src BIGINT,dst BIGINT,paid DOUBLE,
            payment_currency VARCHAR,payment_format VARCHAR,is_laundering BOOLEAN,
            class_name VARCHAR,pair_prior_n BIGINT,paid_currency_z DOUBLE,amt_z_pay_ccy DOUBLE,
            receiver_in_gap_minutes DOUBLE,receiver_in_count BIGINT,pair_prior_count DOUBLE)''')
        con.executemany('INSERT INTO base_input VALUES ('+','.join(['?']*15)+')',targets)
        base = folder/'base.parquet'
        con.execute(f'COPY base_input TO {analysis.q(base)} (FORMAT PARQUET)')
        return folder,tx,base

    def extend(self, folder,tx,base):
        with contextlib.redirect_stdout(io.StringIO()):
            audit = analysis.extend_features(self.con,tx,base,folder/'extended.parquet')
        return self.con.execute('SELECT * FROM extended ORDER BY edge_id').df(),audit

    def test_counts_boundaries_preservation_and_reports(self):
        folder,tx,base = self.fixture('basic')
        df,audit = self.extend(folder,tx,base)
        original = self.con.execute('SELECT * FROM base ORDER BY edge_id').df()
        pd.testing.assert_frame_equal(original,df[original.columns])
        self.assertEqual(df.prior_bucket.tolist(),['0','1','2~5','2~5','6~20','6~20','21+','0','0'])
        for row in df.itertuples():
            others = [r for r in self.rows if r[1]==row.ts and r[0]!=row.edge_id]
            incoming = [r for r in others if r[3]==row.dst]
            self.assertEqual(row.pair_same_minute_other_n,
                             sum(r[2]==row.src and r[3]==row.dst for r in others))
            self.assertEqual(row.receiver_same_minute_other_in_n,len(incoming))
            self.assertEqual(row.receiver_same_minute_other_in_neighbors,len({r[2] for r in incoming}))
            self.assertEqual(row.receiver_same_minute_other_out_n,sum(r[2]==row.dst for r in others))
        indexed = df.set_index('edge_id')
        self.assertEqual(indexed.loc[1000,'pair_same_minute_other_n'],1) # pattern row
        self.assertEqual(indexed.loc[1000,'receiver_same_minute_other_in_neighbors'],4)
        self.assertEqual(indexed.loc[1007,'receiver_same_minute_other_out_n'],1) # self minus one
        self.assertTrue(pd.isna(indexed.loc[1000,'receiver_in_gap_minutes']))
        self.assertEqual(audit['feature_rows'],9)
        with contextlib.redirect_stdout(io.StringIO()):
            tables = analysis.summarize(self.con,folder)
        self.assertEqual(tables['bucket_rates'].total_n.sum(),9)
        zero = tables['bucket_rates'].query("prior_bucket == '0'").iloc[0]
        self.assertEqual(zero.nonpattern_n,1)
        self.assertAlmostEqual(zero.nonpattern_rate,1/3)
        for _,part in tables['same_minute_rates'].groupby('feature'):
            self.assertEqual(part.total_n.sum(),9)
        self.assertEqual(len(list(folder.glob('*.csv'))),8)
        with contextlib.redirect_stdout(io.StringIO()):
            analysis.plot_results(tables,folder)
        self.assertEqual(len(list(folder.glob('*.png'))),3)

    def test_future_and_label_independence(self):
        before,_ = self.extend(*self.fixture('before'))
        future = (3000,datetime(2022,8,8,10,1),1,2,999.,'USD','ACH',True)
        after,_ = self.extend(*self.fixture('future',self.rows+[future]))
        pd.testing.assert_frame_equal(before,after)
        changed = [(*r[:-1],not r[-1]) for r in self.rows]
        flipped,_ = self.extend(*self.fixture('labels',changed))
        cols = ['edge_id','pair_prior_n','receiver_in_gap_minutes']+analysis.MINUTE_METRICS
        pd.testing.assert_frame_equal(before[cols],flipped[cols])

    def test_missing_buckets_and_input_mismatch(self):
        folder,tx,base = self.fixture('empty_bins')
        self.con.execute('DELETE FROM base_input WHERE edge_id<>1000')
        base2 = folder/'base_subset.parquet'
        self.con.execute(f'COPY base_input TO {analysis.q(base2)} (FORMAT PARQUET)')
        self.extend(folder,tx,base2)
        tables = analysis.summarize(self.con,folder)
        empty = tables['bucket_rates'].query('total_n == 0')
        self.assertEqual(len(empty),4)
        self.assertTrue(empty.nonpattern_rate.isna().all())
        # Same edge ID but different account must be rejected before current-row subtraction.
        folder,tx,base = self.fixture('mismatch')
        self.con.execute('UPDATE raw_input SET dst=999 WHERE edge_id=1000')
        badtx = folder/'bad_transactions.parquet'
        self.con.execute(f'COPY raw_input TO {analysis.q(badtx)} (FORMAT PARQUET)')
        with self.assertRaisesRegex(ValueError,'다른 실행'):
            self.extend(folder,badtx,base)

    def test_notebook_is_standalone_and_executes(self):
        import matplotlib
        matplotlib.use('Agg')
        folder,tx,base = self.fixture('notebook')
        project = folder/'project'
        ledger_dir = project/'parquets/HI-Large'
        source_dir = ledger_dir/'nonpaten_vs_normal/pair_history_pit_v1'
        source_dir.mkdir(parents=True)
        import shutil
        shutil.copy2(tx,ledger_dir/'transactions.parquet')
        shutil.copy2(base,source_dir/'features.parquet')
        metadata = dict(status='features_and_tables_complete',feature_rows=9,normal_rows=5,
            nonpattern_rows=4,history_policy='all observed transactions including patterns, strictly before target minute',
            inputs={str(tx):{'bytes':tx.stat().st_size}})
        (source_dir/'analysis_summary.json').write_text(json.dumps(metadata),encoding='utf-8')
        notebook = json.loads(Path(__file__).with_name('03_pair_bins_same_minute_analysis.ipynb').read_text(encoding='utf-8'))
        namespace = {'__name__':'__notebook__'}
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells']:
                if cell['cell_type']!='code':
                    continue
                src = ''.join(cell['source'])
                if 'analysis_functions' in cell.get('metadata',{}).get('tags',[]):
                    self.assertEqual(src,Path(analysis.__file__).read_text(encoding='utf-8'))
                src = src.replace('PROJECT_ROOT = None',f'PROJECT_ROOT = Path({str(project)!r})')
                src = src.replace('REUSE_EXISTING_FEATURES = True','REUSE_EXISTING_FEATURES = False')
                exec(compile(src,'notebook_cell','exec'),namespace)
        output = ledger_dir/'nonpaten_vs_normal/pair_bins_same_minute_until_1105'
        self.assertTrue((output/'analysis_summary.json').exists())
        self.assertFalse((output/'work.duckdb').exists())
        self.assertEqual(len(list(output.glob('*.png'))),3)

    def test_period_boundaries_preserve_features_for_both_classes(self):
        folder,tx,base = self.fixture('period')
        self.extend(folder,tx,base)
        # Deliberately retain the precomputed features; only the target date changes.
        dates = [(3000,1000,'2022-08-07 23:59:00'),
                 (3001,1000,'2022-11-05 23:59:00'),
                 (3002,1001,'2022-11-05 23:59:00'),
                 (3003,1000,'2022-11-06 00:00:00'),
                 (3004,1001,'2022-11-06 00:00:00'),
                 (3005,1000,'2022-11-30 23:59:00')]
        sql = 'SELECT * FROM extended'
        for edge_id,source_id,ts in dates:
            sql += f" UNION ALL SELECT * REPLACE ({edge_id}::BIGINT AS edge_id, TIMESTAMP '{ts}' AS ts) FROM extended WHERE edge_id={source_id}"
        path = folder/'all_periods.parquet'
        self.con.execute(f'COPY ({sql}) TO {analysis.q(path)} (FORMAT PARQUET)')
        audit = analysis.select_analysis_period(self.con,path,'2022-08-08','2022-11-06')
        observed = self.con.execute('SELECT * FROM extended ORDER BY edge_id').df()
        expected = self.con.execute(f'SELECT * FROM read_parquet({analysis.q(path)}) ORDER BY edge_id').df()
        expected = expected[expected.edge_id.isin(self.target_ids+[3001,3002])].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected,observed)
        self.assertEqual(audit['analysis_rows'],11)
        self.assertEqual(audit['normal_rows'],6)
        self.assertEqual(audit['nonpattern_rows'],5)
        self.assertEqual(audit['analysis_last'],'2022-11-05 23:59:00')
        tables = analysis.summarize(self.con,folder)
        nov = tables['monthly_coverage'].query("month == '2022-11'")
        self.assertEqual(set(nov.class_name),{'normal','nonpattern'})
        self.assertTrue((nov.observed_days==1).all())
        self.assertEqual(tables['bucket_rates'].total_n.sum(),11)
        with self.assertRaisesRegex(ValueError,'시작'):
            analysis.select_analysis_period(self.con,path,'2022-11-06','2022-11-06')

    def test_notebook_reuses_features_without_ledger_or_recomputation(self):
        import shutil
        folder,tx,base = self.fixture('reuse')
        self.extend(folder,tx,base)
        project = folder/'project'
        source_dir = project/'parquets/HI-Large/nonpaten_vs_normal/pair_bins_same_minute_v1'
        source_dir.mkdir(parents=True)
        shutil.copy2(folder/'extended.parquet',source_dir/'features.parquet')
        original_bytes = (source_dir/'features.parquet').read_bytes()
        metadata = dict(status='features_and_tables_complete',feature_rows=9,
            prior_policy='02 features unchanged; strictly before target minute',
            same_minute_policy='all labels including patterns; exclude current edge; end-of-minute only')
        (source_dir/'analysis_summary.json').write_text(json.dumps(metadata),encoding='utf-8')
        notebook = json.loads(Path(__file__).with_name('03_pair_bins_same_minute_analysis.ipynb').read_text(encoding='utf-8'))
        namespace = {'__name__':'__notebook__'}
        def fail_if_recomputed(*args,**kwargs):
            self.fail('Reuse mode must not rebuild minute or history features')
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook['cells']:
                if cell['cell_type']!='code':continue
                src = ''.join(cell['source']).replace('PROJECT_ROOT = None',f'PROJECT_ROOT = Path({str(project)!r})')
                exec(compile(src,'notebook_cell','exec'),namespace)
                if 'analysis_functions' in cell.get('metadata',{}).get('tags',[]):
                    namespace['extend_features'] = fail_if_recomputed
        output = source_dir.parent/'pair_bins_same_minute_until_1105'
        audit = json.loads((output/'analysis_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['analysis_end_exclusive'],'2022-11-06')
        self.assertTrue(audit['reused_features'])
        self.assertEqual(audit['analysis_rows'],9)
        self.assertFalse((output/'features.parquet').exists())
        self.assertEqual(len(list(output.glob('*.csv'))),8)
        self.assertEqual(len(list(output.glob('*.png'))),3)
        self.assertEqual((source_dir/'features.parquet').read_bytes(),original_bytes)


if __name__ == '__main__':
    unittest.main()
