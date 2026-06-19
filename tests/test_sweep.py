# -*- coding: utf-8 -*-
import csv
import os
import tempfile
import unittest

from pybaram.api.sweep import (
    _run_aoa_case,
    aoa_case_name,
    collect_force_summary,
    parse_sweep_range,
    parse_sweep_values,
    prepare_case_dir,
    write_sweep_summary
)


class FakeComm:
    rank = 0

    def Barrier(self):
        pass

    def bcast(self, value, root=0):
        return value


class FakeMesh:
    def __init__(self, meshf):
        self.meshf = meshf
        self.closed = False

    def close(self):
        self.closed = True


class SweepValuesTest(unittest.TestCase):
    def test_parse_sweep_values(self):
        self.assertEqual(parse_sweep_values('0, 2.5, -1'), [0.0, 2.5, -1.0])

    def test_parse_sweep_values_rejects_empty(self):
        with self.assertRaises(ValueError):
            parse_sweep_values(' , ')

    def test_parse_positive_range(self):
        self.assertEqual(parse_sweep_range(0, 4, 2), [0.0, 2.0, 4.0])

    def test_parse_negative_range(self):
        self.assertEqual(parse_sweep_range(4, 0, -2), [4.0, 2.0, 0.0])

    def test_parse_range_rejects_wrong_sign(self):
        with self.assertRaises(ValueError):
            parse_sweep_range(0, 4, -1)

    def test_case_name_is_path_safe(self):
        self.assertEqual(aoa_case_name(2), 'aoa2')
        self.assertEqual(aoa_case_name(-2.5), 'aoan2p5')


class SweepSummaryTest(unittest.TestCase):
    def test_collect_force_summary_uses_last_force_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, 'force_airfoil.csv')
            with open(fname, 'w', newline='') as outf:
                writer = csv.writer(outf)
                writer.writerow(['iter', 'cl_p', 'cd_p', 'cmz'])
                writer.writerow([50, 0.1, 0.01, -0.02])
                writer.writerow([100, 0.2, 0.03, -0.04])

            rows = collect_force_summary(tmp, 2.5)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['aoa'], '2.5')
            self.assertEqual(rows[0]['force_file'], 'force_airfoil.csv')
            self.assertEqual(rows[0]['iter'], '100')
            self.assertEqual(rows[0]['cl_p'], '0.2')

    def test_write_sweep_summary_uses_union_of_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, 'sweep.csv')
            write_sweep_summary(fname, [
                {'aoa': '0', 'case': 'aoa0', 'force_file': '', 'status': 'ok'},
                {'aoa': '1', 'case': 'aoa1', 'force_file': 'force.csv',
                 'cl_p': '0.1'}
            ])

            with open(fname, newline='') as inf:
                rows = list(csv.DictReader(inf))

            self.assertEqual(rows[0]['status'], 'ok')
            self.assertEqual(rows[1]['cl_p'], '0.1')


class SweepCaseDirTest(unittest.TestCase):
    def test_prepare_case_dir_rejects_existing_nonempty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = os.path.join(tmp, 'aoa0')
            os.makedirs(case_dir)
            with open(os.path.join(case_dir, 'old.csv'), 'w') as outf:
                outf.write('old')

            with self.assertRaises(RuntimeError):
                prepare_case_dir(case_dir)

    def test_prepare_case_dir_can_overwrite_existing_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = os.path.join(tmp, 'aoa0')
            os.makedirs(case_dir)
            with open(os.path.join(case_dir, 'old.csv'), 'w') as outf:
                outf.write('old')

            prepare_case_dir(case_dir, overwrite=True)

            self.assertTrue(os.path.isdir(case_dir))
            self.assertEqual(os.listdir(case_dir), [])

    def test_run_aoa_case_disables_solver_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_cfg = os.path.join(tmp, 'base.ini')
            with open(base_cfg, 'w') as outf:
                outf.write('[constants]\naoa = 0\n')

            calls = []

            def fake_run(mesh, cfg, comm, ui):
                calls.append((mesh, cfg, comm, ui))
                with open('force_airfoil.csv', 'w', newline='') as outf:
                    writer = csv.writer(outf)
                    writer.writerow(['iter', 'cl_p'])
                    writer.writerow([1, 0.25])

            rows = []
            _run_aoa_case(
                os.path.join(tmp, 'mesh.pbrm'),
                base_cfg,
                tmp,
                os.getcwd(),
                2,
                FakeComm(),
                False,
                rows,
                fake_run,
                FakeMesh
            )

            self.assertEqual(calls[0][3], 'none')
            self.assertEqual(rows[0]['case'], 'aoa2')
            self.assertEqual(rows[0]['cl_p'], '0.25')
            self.assertTrue(os.path.exists(os.path.join(tmp, 'aoa2', 'config.ini')))


if __name__ == '__main__':
    unittest.main()
