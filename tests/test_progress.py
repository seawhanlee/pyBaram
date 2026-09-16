# -*- coding: utf-8 -*-
import io
import unittest
from unittest.mock import patch

from contextlib import redirect_stdout

from pybaram.api.progress import (
    NullProgressHandler,
    RichProgressHandler,
    _HalfSplit,
    _format_remaining,
    _split_widths,
    add_progress_handler,
    progress_snapshot
)
from pybaram.integrators.steady import BaseSteadyIntegrator


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
    def test_default_uses_rich(self):
        intg = FakeIntegrator()
        with patch('pybaram.api.progress.RichProgressHandler') as rich:
            handler = add_progress_handler(intg, FakeComm(0))
        self.assertIs(handler, rich.return_value)
        self.assertEqual(intg.completed_handler, [handler])

    def test_none_ui_does_not_append_handler(self):
        intg = FakeIntegrator()

        handler = add_progress_handler(intg, FakeComm(0), 'none')

        self.assertIsInstance(handler, NullProgressHandler)
        self.assertEqual(intg.completed_handler, [])

    def test_non_root_rank_does_not_append_handler(self):
        intg = FakeIntegrator()

        handler = add_progress_handler(intg, FakeComm(1), 'rich')

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


class RichOutputTest(unittest.TestCase):
    def make_integrator(self, mode='steady'):
        intg = FakeIntegrator()
        intg.mode = mode
        intg.iter = 3
        intg.itermax = 10
        intg.tcurr = 0.25
        intg.tend = 1.0
        intg.resid = FakeVector([0.5])
        intg.resid0 = FakeVector([1.0])
        intg._res_idx = 0
        intg.conservars = ['rho']
        return intg

    def test_nonterminal_prints_final_status_once_for_each_mode(self):
        from rich.console import Console

        for mode in ('steady', 'unsteady', 'unsteady-dts'):
            with self.subTest(mode=mode):
                out = io.StringIO()
                console = Console(file=out, force_terminal=False, width=120)
                intg = self.make_integrator(mode)
                with patch('rich.console.Console', return_value=console):
                    handler = add_progress_handler(intg, FakeComm(0))
                task = handler._progress.tasks[0]
                self.assertEqual(task.completed, 3 if mode == 'steady' else 0.25)
                handler.start()
                intg.iter = 4
                intg.tcurr = 0.5
                handler(intg)
                self.assertEqual(out.getvalue(), '')
                handler.stop()
                final = out.getvalue()
                self.assertIn(mode, final)
                self.assertIn('4/10' if mode == 'steady' else '0.5/1', final)
                self.assertNotIn('\x1b', final)
                handler.stop()
                self.assertEqual(out.getvalue(), final)

    def test_sweep_preserves_aoa_and_residual_in_final_output(self):
        from rich.console import Console
        from pybaram.api.sweep_progress import SweepProgressContext

        out = io.StringIO()
        context = SweepProgressContext([2, 4])
        context.start_case(2, 0)
        intg = self.make_integrator()
        with patch('rich.console.Console', return_value=Console(
                file=out, force_terminal=False, width=160)):
            handler = add_progress_handler(intg, FakeComm(0), context=context)
        handler.start()
        handler(intg)
        handler.complete_context(intg)
        handler.stop()
        self.assertEqual(context.completed, 1)
        self.assertEqual(context.rows[0], ('2', 'rho = 0.5'))
        for text in ('AOA sweep', 'current aoa', 'rho = 0.5', '1/2'):
            self.assertIn(text, out.getvalue())
        self.assertIsNone(handler._key_reader)

    def test_terminal_updates_inline_and_cleans_up(self):
        from rich.console import Console

        out = io.StringIO()
        intg = self.make_integrator()
        with patch('rich.console.Console', return_value=Console(
                file=out, force_terminal=True, width=100)):
            handler = add_progress_handler(intg, FakeComm(0))
        try:
            handler.start()
            intg.iter = 5
            handler(intg)
            handler._live.refresh()
            self.assertIn('5/10', out.getvalue())
        finally:
            handler.stop()
        self.assertFalse(handler._live.is_started)
        self.assertNotIn('\x1b[?1049h', out.getvalue())


class FinalStatusOutputTest(unittest.TestCase):
    def test_final_status_can_be_suppressed(self):
        intg = BaseSteadyIntegrator.__new__(BaseSteadyIntegrator)
        intg._suppress_final_status = True
        intg._res_idx = 0
        intg.conservars = ['rho']
        intg.tol = 1e-8

        out = io.StringIO()
        with redirect_stdout(out):
            intg.print_res([1e-10])

        self.assertEqual(out.getvalue(), '')


class SweepCLILayoutTest(unittest.TestCase):
    def test_half_split_splits_width_exactly(self):
        self.assertEqual(_split_widths(80), (40, 40))
        self.assertEqual(_split_widths(81), (40, 41))

    def test_half_split_renders_each_side_at_fixed_half_width(self):
        try:
            from rich.console import Console
        except ImportError:
            self.skipTest('rich is not installed')

        class WidthRecorder:
            def __init__(self, text):
                self.text = text
                self.widths = []

            def __rich_console__(self, console, options):
                self.widths.append(options.max_width)
                yield self.text

        left = WidthRecorder('status')
        right = WidthRecorder('residual')
        console = Console(width=81, force_terminal=True)

        console.render_lines(
            _HalfSplit(left, right),
            console.options.update(width=81, max_width=81),
            pad=True
        )

        self.assertEqual(left.widths, [40])
        self.assertEqual(right.widths, [41])

    def test_sweep_layout_uses_half_split(self):
        class Context:
            rows = [('0', 'pending')]

        handler = RichProgressHandler.__new__(RichProgressHandler)
        handler._context = Context()

        grid = handler._sweep_layout('left status area')

        self.assertIsInstance(grid, _HalfSplit)


if __name__ == '__main__':
    unittest.main()
