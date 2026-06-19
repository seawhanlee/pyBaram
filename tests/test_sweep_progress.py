# -*- coding: utf-8 -*-
import unittest

from pybaram.api.sweep_progress import (
    NullSweepProgress,
    make_sweep_progress
)


class FakeComm:
    def __init__(self, rank):
        self.rank = rank


class SweepProgressTest(unittest.TestCase):
    def test_none_ui_returns_null_progress(self):
        progress = make_sweep_progress([0, 1], FakeComm(0), 'none')

        self.assertIsInstance(progress, NullSweepProgress)

    def test_non_root_rank_returns_null_progress(self):
        progress = make_sweep_progress([0, 1], FakeComm(1), 'tui')

        self.assertIsInstance(progress, NullSweepProgress)

    def test_invalid_ui_raises(self):
        with self.assertRaises(ValueError):
            make_sweep_progress([0, 1], FakeComm(0), 'bad-ui')


if __name__ == '__main__':
    unittest.main()
