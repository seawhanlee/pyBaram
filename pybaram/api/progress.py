# -*- coding: utf-8 -*-
import sys

from time import perf_counter


_UI_CHOICES = {'tqdm', 'tui', 'none'}


def add_progress_handler(integrator, comm, ui='tqdm', context=None):
    if ui not in _UI_CHOICES:
        raise ValueError("Unknown progress UI {!r}".format(ui))

    if ui == 'none' or getattr(comm, 'rank', 0) != 0:
        return NullProgressHandler()

    handler = (
        RichProgressHandler(integrator, context)
        if ui == 'tui'
        else TqdmProgressHandler(integrator, context)
    )
    integrator.completed_handler.append(handler)

    return handler


def progress_snapshot(intg):
    mode = getattr(intg, 'mode', 'unknown')

    if mode in ('unsteady', 'unsteady-dts'):
        total = _time_total(intg)
        completed = min(getattr(intg, 'tcurr', 0.0), total)
    else:
        total = getattr(intg, 'itermax', getattr(intg, 'iter', 0))
        completed = min(getattr(intg, 'iter', 0), total)

    return {
        'mode': mode,
        'total': total,
        'completed': completed,
        'rows': _progress_rows(intg, total)
    }


class NullProgressHandler:
    def start(self):
        pass

    def stop(self):
        pass

    def __call__(self, intg):
        pass

    def complete_context(self, intg):
        pass


class TqdmProgressHandler:
    def __init__(self, intg, context=None):
        from tqdm import tqdm

        self._context = context
        snap = progress_snapshot(intg)
        self._completed = snap['completed']
        self._bar = tqdm(
            total=snap['total'],
            initial=snap['completed'],
            unit_scale=snap['mode'] in ('unsteady', 'unsteady-dts'),
            leave=context is None
        )

    def start(self):
        pass

    def stop(self):
        self._bar.close()

    def complete_context(self, intg):
        if self._context is not None:
            self._context.complete_case()
            self._set_postfix()

    def __call__(self, intg):
        completed = progress_snapshot(intg)['completed']
        update = max(completed - self._completed, 0)
        self._completed += update
        self._bar.update(update)
        self._set_postfix()

    def _set_postfix(self):
        if self._context is None:
            return

        self._bar.set_postfix(
            aoa=self._context.current,
            sweep='{}/{}'.format(self._context.completed, self._context.total)
        )


class RichProgressHandler:
    def __init__(self, intg, context=None):
        self._context = context
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
                "--ui tui requires the rich package; progress display disabled."
            )
            return

        self._console = Console(stderr=True)
        if not self._console.is_terminal:
            self._disabled = True
            self._message = (
                "--ui tui requires an interactive terminal; progress display "
                "disabled."
            )
            return

        snap = progress_snapshot(intg)
        self._sweep_progress = None
        self._sweep_task = None
        if context is not None:
            self._sweep_progress = Progress(
                TextColumn("[bold]AOA sweep[/bold]"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self._console,
                transient=context is not None
            )
            self._sweep_task = self._sweep_progress.add_task(
                'sweep',
                total=context.total,
                completed=context.completed
            )

        self._progress = Progress(
            TextColumn("[bold]pyBaram[/bold]"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
            transient=context is not None
        )
        self._task = self._progress.add_task(
            'simulation',
            total=snap['total'],
            completed=snap['completed']
        )
        self._live = Live(
            self._render(intg),
            console=self._console,
            refresh_per_second=4,
            transient=context is not None
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

    def __call__(self, intg):
        if self._disabled:
            return

        snap = progress_snapshot(intg)
        self._update_context_progress()
        self._progress.update(self._task, completed=snap['completed'])
        self._live.update(self._render(intg))

    def complete_context(self, intg):
        if self._disabled or self._context is None:
            return

        self._context.complete_case()
        self._update_context_progress()
        self._live.update(self._render(intg))

    def _update_context_progress(self):
        if self._context is None:
            return

        self._sweep_progress.update(
            self._sweep_task,
            completed=self._context.completed
        )

    def _render(self, intg):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        snap = progress_snapshot(intg)
        rows = snap['rows']

        table = Table.grid(padding=(0, 2))
        table.add_column(style='cyan', no_wrap=True)
        table.add_column()

        for name, value in rows:
            table.add_row(name, value)

        elapsed = perf_counter() - self._start_time
        if self._context is not None:
            table.add_row('current aoa', self._context.current)
            table.add_row(
                'sweeps',
                '{}/{}'.format(self._context.completed, self._context.total)
            )

        table.add_row('elapsed', _format_seconds(elapsed))
        table.add_row('remaining', _format_remaining(
            elapsed,
            snap['completed'],
            snap['total']
        ))

        items = []
        if self._sweep_progress is not None:
            items.append(self._sweep_progress)
        items.extend([self._progress, table])

        return Panel(
            Group(*items),
            title='pyBaram simulation',
            border_style='blue'
        )


def _time_total(intg):
    if hasattr(intg, 'tend'):
        return intg.tend

    return intg.tlist[-1]


def _progress_rows(intg, total):
    mode = getattr(intg, 'mode', 'unknown')
    rows = [('mode', mode)]

    if mode == 'steady':
        rows.append((
            'iteration',
            '{}/{}'.format(getattr(intg, 'iter', 0), getattr(intg, 'itermax', 0))
        ))
        residual = _steady_residual(intg)
        if residual is not None:
            rows.append(('residual', residual))
        if hasattr(intg, 'tol'):
            rows.append(('tolerance', _format_float(intg.tol)))
        if hasattr(intg, 'cfl'):
            rows.append(('cfl', _format_float(intg.cfl)))
    elif mode == 'unsteady-dts':
        rows.append(('time', '{}/{}'.format(
            _format_float(getattr(intg, 'tcurr', 0.0)),
            _format_float(total)
        )))
        if hasattr(intg, 'piter'):
            rows.append(('physical step', str(intg.piter)))
        rows.append(('pseudo iteration', str(getattr(intg, 'iter', 0))))
        if hasattr(intg, 'subitnum'):
            rows.append(('last subiter', str(intg.subitnum)))
        if hasattr(intg, 'subres'):
            rows.append(('subres', _format_float(intg.subres)))
        if hasattr(intg, 'subtol'):
            rows.append(('sub tolerance', _format_float(intg.subtol)))
        if hasattr(intg, 'scfl'):
            rows.append(('sub cfl', _format_float(intg.scfl)))
    else:
        rows.append(('time', '{}/{}'.format(
            _format_float(getattr(intg, 'tcurr', 0.0)),
            _format_float(total)
        )))
        rows.append(('iteration', str(getattr(intg, 'iter', 0))))
        if hasattr(intg, 'dt'):
            rows.append(('dt', _format_float(intg.dt)))
        if hasattr(intg, 'cfl'):
            rows.append(('cfl', _format_float(intg.cfl)))

    return rows


def _steady_residual(intg):
    try:
        resid = intg.resid / intg.resid0
        idx = intg._res_idx
        name = intg.conservars[idx]
        return '{} = {}'.format(name, _format_float(resid[idx]))
    except (AttributeError, IndexError, TypeError, ZeroDivisionError):
        return None


def _format_float(value):
    try:
        return '{:.6g}'.format(value)
    except (TypeError, ValueError):
        return str(value)


def _format_seconds(seconds):
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)

    if hours:
        return '{}h {}m {}s'.format(hours, minute, sec)
    if minute:
        return '{}m {}s'.format(minute, sec)

    return '{}s'.format(sec)


def _format_remaining(elapsed, completed, total):
    try:
        elapsed = float(elapsed)
        completed = float(completed)
        total = float(total)
    except (TypeError, ValueError):
        return 'unknown'

    if total <= 0:
        return 'unknown'
    if completed >= total:
        return '0s'
    if completed <= 0 or elapsed <= 0:
        return 'estimating'

    remaining = elapsed * (total - completed) / completed
    return _format_seconds(remaining)
