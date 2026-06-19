# -*- coding: utf-8 -*-
import unittest

from pybaram.api.progress import (
    NullProgressHandler,
    _format_remaining,
    add_progress_handler,
    progress_snapshot
)


class FakeComm:
    def __init__(self, rank):
        self.rank = rank


class FakeIntegrator:
    def __init__(self):
        self.completed_handler = []


class FakeVector:
    def __init__(self, values):
        self._values = values

    def __truediv__(self, other):
        return FakeVector([
            value / denom
            for value, denom in zip(self._values, other._values)
        ])

    def __getitem__(self, idx):
        return self._values[idx]


class ProgressSnapshotTest(unittest.TestCase):
    def test_steady_snapshot_includes_residual(self):
        intg = FakeIntegrator()
        intg.mode = 'steady'
        intg.iter = 3
        intg.itermax = 10
        intg.resid = FakeVector([2.0, 5.0])
        intg.resid0 = FakeVector([4.0, 5.0])
        intg._res_idx = 0
        intg.conservars = ['rho', 'rhou']
        intg.tol = 1e-8
        intg.cfl = 2.5

        snap = progress_snapshot(intg)
        rows = dict(snap['rows'])

        self.assertEqual(snap['mode'], 'steady')
        self.assertEqual(snap['total'], 10)
        self.assertEqual(snap['completed'], 3)
        self.assertEqual(rows['iteration'], '3/10')
        self.assertEqual(rows['residual'], 'rho = 0.5')

    def test_unsteady_snapshot_uses_physical_time(self):
        intg = FakeIntegrator()
        intg.mode = 'unsteady'
        intg.tcurr = 0.25
        intg.tlist = [0.0, 0.5, 1.0]
        intg.iter = 4
        intg.dt = 0.05

        snap = progress_snapshot(intg)
        rows = dict(snap['rows'])

        self.assertEqual(snap['total'], 1.0)
        self.assertEqual(snap['completed'], 0.25)
        self.assertEqual(rows['time'], '0.25/1')
        self.assertEqual(rows['iteration'], '4')

    def test_dts_snapshot_uses_tend(self):
        intg = FakeIntegrator()
        intg.mode = 'unsteady-dts'
        intg.tcurr = 0.4
        intg.tend = 1.0
        intg.iter = 20
        intg.piter = 4
        intg.subitnum = 5
        intg.subres = 1e-4
        intg.subtol = 1e-3
        intg.scfl = 10.0

        snap = progress_snapshot(intg)
        rows = dict(snap['rows'])

        self.assertEqual(snap['total'], 1.0)
        self.assertEqual(snap['completed'], 0.4)
        self.assertEqual(rows['physical step'], '4')
        self.assertEqual(rows['subres'], '0.0001')


class AddProgressHandlerTest(unittest.TestCase):
    def test_none_ui_does_not_append_handler(self):
        intg = FakeIntegrator()

        handler = add_progress_handler(intg, FakeComm(0), 'none')

        self.assertIsInstance(handler, NullProgressHandler)
        self.assertEqual(intg.completed_handler, [])

    def test_non_root_rank_does_not_append_handler(self):
        intg = FakeIntegrator()

        handler = add_progress_handler(intg, FakeComm(1), 'tqdm')

        self.assertIsInstance(handler, NullProgressHandler)
        self.assertEqual(intg.completed_handler, [])

    def test_invalid_ui_raises(self):
        intg = FakeIntegrator()

        with self.assertRaises(ValueError):
            add_progress_handler(intg, FakeComm(0), 'bad-ui')


class RemainingTimeTest(unittest.TestCase):
    def test_remaining_time_is_estimated_from_progress(self):
        self.assertEqual(_format_remaining(30, 25, 100), '1m 30s')

    def test_remaining_time_is_zero_when_complete(self):
        self.assertEqual(_format_remaining(30, 100, 100), '0s')

    def test_remaining_time_is_estimating_without_progress(self):
        self.assertEqual(_format_remaining(30, 0, 100), 'estimating')

    def test_remaining_time_is_unknown_without_total(self):
        self.assertEqual(_format_remaining(30, 0, 0), 'unknown')


if __name__ == '__main__':
    unittest.main()
