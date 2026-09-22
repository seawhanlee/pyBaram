# -*- coding: utf-8 -*-
import os
import stat
import sys

from pathlib import Path
from time import perf_counter


_UI_CHOICES = {"rich", "none"}


def _mpi_terminal_width():
    """Find a local MPI launcher's terminal when rank stderr is a pipe.

    MPICH/Hydra forwards rank output through pipes even in interactive runs.
    Inspect only the nearest known launcher: a redirected launcher must not
    inherit terminal status from an interactive shell further up the tree.
    Missing /proc access or a remote launcher leaves normal Rich detection in
    charge. Never write directly to the launcher's terminal.
    """
    if sys.platform != 'linux' or os.environ.get('TERM') in ('dumb', 'unknown'):
        return None
    if not any(key in os.environ for key in (
            'PMI_RANK', 'PMIX_RANK', 'OMPI_COMM_WORLD_RANK')):
        return None

    launchers = {'mpirun', 'mpiexec', 'mpiexec.hydra', 'orterun', 'prterun', 'srun'}
    pid = os.getppid()
    try:
        for _ in range(32):
            if pid <= 1:
                break
            proc = Path('/proc') / str(pid)
            if (proc / 'comm').read_text().strip() in launchers:
                stderr = proc / 'fd/2'
                if not stat.S_ISCHR(stderr.stat().st_mode):
                    return None
                fd = os.open(stderr, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
                try:
                    if os.isatty(fd):
                        return os.get_terminal_size(fd).columns or 80
                finally:
                    os.close(fd)
                return None
            pid = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def add_progress_handler(integrator, comm, ui="rich", context=None):
    if ui not in _UI_CHOICES:
        raise ValueError("Unknown progress UI {!r}".format(ui))

    if ui == "none" or getattr(comm, "rank", 0) != 0:
        return NullProgressHandler()

    handler = RichProgressHandler(integrator, context)
    integrator.completed_handler.append(handler)

    return handler


def progress_snapshot(intg):
    mode = getattr(intg, "mode", "unknown")

    if mode in ("unsteady", "unsteady-dts"):
        total = _time_total(intg)
        completed = min(getattr(intg, "tcurr", 0.0), total)
    else:
        total = getattr(intg, "itermax", getattr(intg, "iter", 0))
        completed = min(getattr(intg, "iter", 0), total)

    return {
        "mode": mode,
        "total": total,
        "completed": completed,
        "rows": _progress_rows(intg, total),
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


class RichProgressHandler:
    def __init__(self, intg, context=None):
        self._context = context
        self._integrator = intg
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
                TimeElapsedColumn,
            )
        except ImportError:
            self._disabled = True
            self._message = (
                "--ui rich requires the rich package; progress display disabled."
            )
            return

        self._console = Console(stderr=True)
        if not self._console.is_terminal:
            width = _mpi_terminal_width()
            if width is not None:
                self._console = Console(stderr=True, force_terminal=True, width=width)
        self._interactive = self._console.is_terminal
        self._key_reader = (
            _SweepKeyReader(context)
            if context is not None and self._interactive else None
        )
        snap = progress_snapshot(intg)
        self._initial_iteration = getattr(intg, "iter", 0)
        self._initial_completed = snap["completed"]
        self._sweep_progress = None
        self._sweep_task = None
        if context is not None:
            self._sweep_progress = Progress(
                TextColumn("[bold]AOA sweep[/bold]"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self._console,
                transient=context is not None,
            )
            self._sweep_task = self._sweep_progress.add_task(
                "sweep", total=context.total, completed=context.completed
            )

        self._progress = Progress(
            TextColumn("[bold]pyBaram[/bold]"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("Elapsed [yellow]{task.fields[elapsed]}[/yellow]"),
            TextColumn("[cyan]{task.fields[rate]}[/cyan] it/s"),
            TextColumn("ETA [yellow]{task.fields[eta]}[/yellow]"),
            expand=True,
            console=self._console,
            transient=context is not None,
        )
        self._task = self._progress.add_task(
            "simulation", total=snap["total"], completed=snap["completed"],
            elapsed="0s", rate="--", eta="estimating",
        )
        self._live = Live(
            self._render(intg),
            console=self._console,
            refresh_per_second=4,
            transient=context is not None,
        )

    def start(self):
        if self._disabled:
            if self._message:
                print("[pybaram] {}".format(self._message), file=sys.stderr)
                self._message = None
            return

        if self._interactive:
            self._live.start()
        if self._key_reader is not None:
            self._key_reader.start()
        self._started = True

    def stop(self):
        if self._key_reader is not None:
            self._key_reader.stop()
        if self._started:
            if self._interactive:
                self._live.stop()
            else:
                self._console.print(self._render(self._integrator))
            self._started = False

    def __call__(self, intg):
        self._integrator = intg
        if self._disabled:
            return

        if self._key_reader is not None:
            self._key_reader.poll()
        snap = progress_snapshot(intg)
        if self._context is not None:
            self._context.update_case(_case_residual(intg))
        self._update_context_progress()
        self._progress.update(self._task, completed=snap["completed"])
        self._live.update(self._render(intg))

    def complete_context(self, intg):
        self._integrator = intg
        if self._disabled or self._context is None:
            return

        self._context.complete_case(_case_residual(intg))
        self._update_context_progress()
        self._live.update(self._render(intg))

    def _update_context_progress(self):
        if self._context is None:
            return

        self._sweep_progress.update(self._sweep_task, completed=self._context.completed)

    def _render(self, intg):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        snap = progress_snapshot(intg)
        rows = snap["rows"]

        status_table = Table.grid(padding=(0, 2))
        status_table.add_column(style="cyan", no_wrap=True)
        status_table.add_column()

        for name, value in rows:
            status_table.add_row(name, value)

        elapsed = perf_counter() - self._start_time
        iterations = getattr(intg, "iter", 0) - self._initial_iteration
        self._progress.update(
            self._task,
            elapsed=_format_seconds(elapsed),
            rate="{:.2f}".format(iterations / elapsed)
            if iterations > 0 and elapsed > 0 else "--",
            eta=_format_remaining(
                elapsed, snap["completed"] - self._initial_completed,
                snap["total"] - self._initial_completed,
            ) if snap["completed"] < snap["total"] else "0s",
        )
        if self._context is not None:
            status_table.add_row("current aoa", self._context.current)
            status_table.add_row(
                "sweeps", "{}/{}".format(self._context.completed, self._context.total)
            )
            if self._context.stop_requested:
                status_table.add_row("stop", "requested after current aoa")

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

        return Panel(content, title="pyBaram", border_style="blue")

    def _sweep_layout(self, status_area):
        return _HalfSplit(status_area, self._sweep_residual_table())

    def _sweep_residual_table(self):
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text

        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column()
        table.add_row("AOA", "Residual")

        for aoa, residual in self._context.rows:
            table.add_row(aoa, residual)

        return Group(Text("Sweep residuals", style="bold"), table)


class _HalfSplit:
    def __init__(self, left, right):
        self._left = left
        self._right = right

    def __rich_console__(self, console, options):
        from rich.segment import Segment

        left_width, right_width = _split_widths(options.max_width)
        left_options = options.update(
            width=left_width, min_width=left_width, max_width=left_width
        )
        right_options = options.update(
            width=right_width, min_width=right_width, max_width=right_width
        )
        left_lines = console.render_lines(self._left, left_options, pad=True)
        right_lines = console.render_lines(self._right, right_options, pad=True)
        height = max(len(left_lines), len(right_lines))

        for i in range(height):
            left_line = left_lines[i] if i < len(left_lines) else []
            right_line = right_lines[i] if i < len(right_lines) else []
            line = Segment.adjust_line_length(
                left_line, left_width
            ) + Segment.adjust_line_length(right_line, right_width)

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
            if sys.platform == "win32":
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
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old_attrs)
        self._active = False

    def poll(self):
        if not self._active:
            return

        key = self._read_key()
        if key in ("q", "Q"):
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
    if hasattr(intg, "tend"):
        return intg.tend

    return intg.tlist[-1]


def _progress_rows(intg, total):
    mode = getattr(intg, "mode", "unknown")
    rows = [("mode", mode)]

    if mode == "steady":
        rows.append(
            (
                "iteration",
                "{}/{}".format(getattr(intg, "iter", 0), getattr(intg, "itermax", 0)),
            )
        )
        residual = _steady_residual(intg)
        if residual is not None:
            rows.append(("residual", residual))
        if hasattr(intg, "tol"):
            rows.append(("tolerance", _format_float(intg.tol)))
        if hasattr(intg, "cfl"):
            rows.append(("cfl", _format_float(intg.cfl)))
    elif mode == "unsteady-dts":
        rows.append(
            (
                "time",
                "{}/{}".format(
                    _format_float(getattr(intg, "tcurr", 0.0)), _format_float(total)
                ),
            )
        )
        if hasattr(intg, "piter"):
            rows.append(("physical step", str(intg.piter)))
        rows.append(("pseudo iteration", str(getattr(intg, "iter", 0))))
        if hasattr(intg, "subitnum"):
            rows.append(("last subiter", str(intg.subitnum)))
        if hasattr(intg, "subres"):
            rows.append(("subres", _format_float(intg.subres)))
        if hasattr(intg, "subtol"):
            rows.append(("sub tolerance", _format_float(intg.subtol)))
        if hasattr(intg, "scfl"):
            rows.append(("sub cfl", _format_float(intg.scfl)))
    else:
        rows.append(
            (
                "time",
                "{}/{}".format(
                    _format_float(getattr(intg, "tcurr", 0.0)), _format_float(total)
                ),
            )
        )
        rows.append(("iteration", str(getattr(intg, "iter", 0))))
        if hasattr(intg, "dt"):
            rows.append(("dt", _format_float(intg.dt)))
        if hasattr(intg, "cfl"):
            rows.append(("cfl", _format_float(intg.cfl)))

    return rows


def _steady_residual(intg):
    try:
        resid = intg.resid / intg.resid0
        idx = intg._res_idx
        name = intg.conservars[idx]
        return "{} = {}".format(name, _format_float(resid[idx]))
    except (AttributeError, IndexError, TypeError, ZeroDivisionError):
        return None


def _case_residual(intg):
    residual = _steady_residual(intg)
    if residual is not None:
        return residual

    if hasattr(intg, "subres"):
        return "subres = {}".format(_format_float(intg.subres))

    return None


def _format_float(value):
    try:
        return "{:.6g}".format(value)
    except (TypeError, ValueError):
        return str(value)


def _format_seconds(seconds):
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)

    if hours:
        return "{}h {}m {}s".format(hours, minute, sec)
    if minute:
        return "{}m {}s".format(minute, sec)

    return "{}s".format(sec)


def _format_remaining(elapsed, completed, total):
    try:
        elapsed = float(elapsed)
        completed = float(completed)
        total = float(total)
    except (TypeError, ValueError):
        return "unknown"

    if total <= 0:
        return "unknown"
    if completed >= total:
        return "0s"
    if completed <= 0 or elapsed <= 0:
        return "estimating"

    remaining = elapsed * (total - completed) / completed
    return _format_seconds(remaining)
