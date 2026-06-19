# -*- coding: utf-8 -*-
import sys

from time import perf_counter

from pybaram.api.progress import _format_remaining, _format_seconds
from pybaram.api.sweep import format_sweep_value


_UI_CHOICES = {'tqdm', 'tui', 'none'}


def make_sweep_progress(aoas, comm, ui='tui'):
    if ui not in _UI_CHOICES:
        raise ValueError("Unknown sweep UI {!r}".format(ui))

    if ui == 'none' or getattr(comm, 'rank', 0) != 0:
        return NullSweepProgress()

    handler = (
        RichSweepProgress(aoas)
        if ui == 'tui'
        else TqdmSweepProgress(aoas)
    )

    return handler


class NullSweepProgress:
    def start(self):
        pass

    def stop(self):
        pass

    def start_case(self, aoa, index):
        pass

    def complete_case(self, aoa, index):
        pass


class TqdmSweepProgress:
    def __init__(self, aoas):
        from tqdm import tqdm

        self._bar = tqdm(total=len(aoas))
        self._bar.set_description('AOA sweep')

    def start(self):
        pass

    def stop(self):
        self._bar.close()

    def start_case(self, aoa, index):
        self._bar.set_postfix(aoa=format_sweep_value(aoa))

    def complete_case(self, aoa, index):
        self._bar.update(1)


class RichSweepProgress:
    def __init__(self, aoas):
        self._aoas = list(aoas)
        self._current = 'pending'
        self._completed = 0
        self._started = False
        self._disabled = False
        self._message = None
        self._start_time = perf_counter()

        try:
            from rich.console import Console
            from rich.live import Live
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn
            )
        except ImportError:
            self._disabled = True
            self._message = (
                "--ui tui requires the rich package; sweep display disabled."
            )
            return

        self._console = Console(stderr=True)
        if not self._console.is_terminal:
            self._disabled = True
            self._message = (
                "--ui tui requires an interactive terminal; sweep display "
                "disabled."
            )
            return

        self._progress = Progress(
            TextColumn("[bold]AOA sweep[/bold]"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=False
        )
        self._task = self._progress.add_task(
            'sweep',
            total=len(self._aoas),
            completed=0
        )
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=False
        )

    def start(self):
        if self._disabled:
            if self._message:
                print("[pybaram] {}".format(self._message), file=sys.stderr)
                self._message = None
            return

        self._live.start()
        self._started = True

    def stop(self):
        if self._started:
            self._live.stop()
            self._started = False

    def start_case(self, aoa, index):
        if self._disabled:
            return

        self._current = format_sweep_value(aoa)
        self._live.update(self._render())

    def complete_case(self, aoa, index):
        if self._disabled:
            return

        self._completed = index + 1
        self._progress.update(self._task, completed=self._completed)
        if self._completed == len(self._aoas):
            self._current = 'complete'
        self._live.update(self._render())

    def _render(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        elapsed = perf_counter() - self._start_time

        table = Table.grid(padding=(0, 2))
        table.add_column(style='cyan', no_wrap=True)
        table.add_column()
        table.add_row('current aoa', self._current)
        table.add_row(
            'completed',
            '{}/{}'.format(self._completed, len(self._aoas))
        )
        table.add_row('elapsed', _format_seconds(elapsed))
        table.add_row('remaining', _format_remaining(
            elapsed,
            self._completed,
            len(self._aoas)
        ))

        return Panel(
            Group(self._progress, table),
            title='pyBaram AOA sweep',
            border_style='blue'
        )
