# -*- coding: utf-8 -*-
import unittest

from pybaram.api.sweep_progress import (
    SweepProgressContext
)


class SweepProgressTest(unittest.TestCase):
    def test_context_tracks_current_case(self):
        context = SweepProgressContext([0, 2])

        context.start_case(2, 1)

        self.assertEqual(context.current, '2')
        self.assertEqual(context.completed, 0)
        self.assertEqual(context.total, 2)

    def test_context_marks_complete(self):
        context = SweepProgressContext([0, 2])

        context.start_case(2, 1)
        context.complete_case()

        self.assertEqual(context.current, 'complete')
        self.assertEqual(context.completed, 2)


if __name__ == '__main__':
    unittest.main()
