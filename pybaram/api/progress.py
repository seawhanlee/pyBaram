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
            self._context.complete_case(_case_residual(intg))
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
            residual=self._current_context_residual(),
            sweep='{}/{}'.format(self._context.completed, self._context.total)
        )

    def _current_context_residual(self):
        for aoa, residual in self._context.rows:
            if aoa == self._context.current:
                return residual

        return ''


class RichProgressHandler:
    def __init__(self, intg, context=None):
        self._context = context
        self._started = False
        self._disabled = False
        self._key_reader = None
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

        self._key_reader = _SweepKeyReader(context) if context is not None else None
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
        if self._key_reader is not None:
            self._key_reader.start()
        self._started = True

    def stop(self):
        if self._key_reader is not None:
            self._key_reader.stop()
        if self._started:
            self._live.stop()
            self._started = False

    def __call__(self, intg):
        if self._disabled:
            return

        if self._key_reader is not None:
            self._key_reader.poll()
        snap = progress_snapshot(intg)
        if self._context is not None:
            self._context.update_case(_case_residual(intg))
        self._update_context_progress()
        self._progress.update(self._task, completed=snap['completed'])
        self._live.update(self._render(intg))

    def complete_context(self, intg):
        if self._disabled or self._context is None:
            return

        self._context.complete_case(_case_residual(intg))
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

        status_table = Table.grid(padding=(0, 2))
        status_table.add_column(style='cyan', no_wrap=True)
        status_table.add_column()

        for name, value in rows:
            status_table.add_row(name, value)

        elapsed = perf_counter() - self._start_time
        if self._context is not None:
            status_table.add_row('current aoa', self._context.current)
            status_table.add_row(
                'sweeps',
                '{}/{}'.format(self._context.completed, self._context.total)
            )
            if self._context.stop_requested:
                status_table.add_row('stop', 'requested after current aoa')

        status_table.add_row('elapsed', _format_seconds(elapsed))
        status_table.add_row('remaining', _format_remaining(
            elapsed,
            snap['completed'],
            snap['total']
        ))

        if self._context is None:
            items = [self._progress, status_table]
            content = Group(*items)
        else:
            left_items = []
            if self._sweep_progress is not None:
                left_items.append(self._sweep_progress)
            left_items.append(self._progress)
            left_items.append(status_table)
            content = self._sweep_layout(Group(*left_items))

        return Panel(
            content,
            title='pyBaram simulation',
            border_style='blue'
        )

    def _sweep_layout(self, status_area):
        return _HalfSplit(status_area, self._sweep_residual_table())

    def _sweep_residual_table(self):
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(style='cyan', no_wrap=True)
        table.add_column()
        table.add_row('AOA', 'Residual')

        for aoa, residual in self._context.rows:
            table.add_row(aoa, residual)

        return Group(Text('Sweep residuals', style='bold'), table)


class _HalfSplit:
    def __init__(self, left, right):
        self._left = left
        self._right = right

    def __rich_console__(self, console, options):
        from rich.segment import Segment

        left_width, right_width = _split_widths(options.max_width)
        left_options = options.update(
            width=left_width,
            min_width=left_width,
            max_width=left_width
        )
        right_options = options.update(
            width=right_width,
            min_width=right_width,
            max_width=right_width
        )
        left_lines = console.render_lines(self._left, left_options, pad=True)
        right_lines = console.render_lines(self._right, right_options, pad=True)
        height = max(len(left_lines), len(right_lines))

        for i in range(height):
            left_line = left_lines[i] if i < len(left_lines) else []
            right_line = right_lines[i] if i < len(right_lines) else []
            line = (
                Segment.adjust_line_length(left_line, left_width) +
                Segment.adjust_line_length(right_line, right_width)
            )

            for segment in line:
                yield segment

            if i + 1 < height:
                yield Segment.line()


def _split_widths(width):
    left_width = width // 2
    return left_width, width - left_width


class _SweepKeyReader:
    def __init__(self, context):
        self._context = context
        self._active = False
        self._fd = None
        self._old_attrs = None
        self._msvcrt = None
        self._select = None
        self._termios = None

    def start(self):
        if not sys.stdin.isatty():
            return

        try:
            if sys.platform == 'win32':
                import msvcrt
                self._msvcrt = msvcrt
            else:
                import select
                import termios
                import tty

                self._fd = sys.stdin.fileno()
                self._old_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self._select = select
                self._termios = termios
        except Exception:
            return

        self._active = True

    def stop(self):
        if not self._active:
            return

        if self._termios is not None and self._old_attrs is not None:
            self._termios.tcsetattr(
                self._fd,
                self._termios.TCSADRAIN,
                self._old_attrs
            )
        self._active = False

    def poll(self):
        if not self._active:
            return

        key = self._read_key()
        if key in ('q', 'Q'):
            self._context.request_stop()

    def _read_key(self):
        if self._msvcrt is not None:
            if self._msvcrt.kbhit():
                return self._msvcrt.getwch()
            return None

        readable, _, _ = self._select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.read(1)

        return None


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


def _case_residual(intg):
    residual = _steady_residual(intg)
    if residual is not None:
        return residual

    if hasattr(intg, 'subres'):
        return 'subres = {}'.format(_format_float(intg.subres))

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
