# -*- coding: utf-8 -*-
"""Interactive Rich terminal launcher for pyBaram workflows.

The solver progress dashboard lives in :mod:`pybaram.api.progress`.  This
module provides the missing "front door" TUI: a guided terminal experience for
building and launching every public pyBaram command without memorising CLI
syntax.
"""
import os
import shlex

from dataclasses import dataclass


_UI_CHOICES = ('tui', 'tqdm', 'none')


@dataclass(frozen=True)
class CommandPreview:
    """Command assembled by the interactive launcher."""

    title: str
    argv: tuple

    @property
    def shell_command(self):
        return shlex.join(('pybaram',) + self.argv)


def build_run_command(mesh, ini, ui='tui'):
    return CommandPreview(
        'Run simulation',
        ('run', mesh, ini, '--ui', _validate_ui(ui))
    )


def build_restart_command(mesh, soln, ini=None, ui='tui'):
    argv = ['restart', mesh, soln]
    if ini:
        argv.append(ini)
    argv.extend(('--ui', _validate_ui(ui)))
    return CommandPreview('Restart simulation', tuple(argv))


def build_sweep_command(mesh, ini, aoa_values=None, aoa_range=None,
                        out='sweep-aoa', ui='tui', overwrite=False,
                        resume=False):
    if bool(aoa_values) == bool(aoa_range):
        raise ValueError('Provide exactly one of aoa_values or aoa_range')
    if overwrite and resume:
        raise ValueError('overwrite and resume cannot both be enabled')

    argv = ['sweep', mesh, ini]
    if aoa_values:
        argv.extend(('--aoa', aoa_values))
    else:
        start, stop, step = aoa_range
        argv.extend(('--aoa-range', start, stop, step))

    if out:
        argv.extend(('--out', out))
    argv.extend(('--ui', _validate_ui(ui)))
    if overwrite:
        argv.append('--overwrite')
    if resume:
        argv.append('--resume')

    return CommandPreview('AOA sweep', tuple(argv))


def build_import_command(inmesh, outmesh, scale=1.0):
    return CommandPreview(
        'Import mesh',
        ('import', inmesh, outmesh, '--scale', _format_number(scale))
    )


def build_partition_command(npart, mesh, out, soln=()):
    argv = ['partition', str(npart), mesh]
    argv.extend(soln or ())
    argv.append(out)
    return CommandPreview('Partition mesh', tuple(argv))


def build_export_command(mesh, soln=None, out=None, surface=None,
                         list_surfaces=False):
    argv = ['export', mesh]
    if soln:
        argv.append(soln)
    if out:
        argv.append(out)
    if surface:
        argv.extend(('--surface', surface))
    if list_surfaces:
        argv.append('--list-surfaces')
    return CommandPreview('Export solution', tuple(argv))


class PyBaramTUI:
    """Rich-powered interactive launcher for all pyBaram workflows."""

    def __init__(self, console=None, executor=None):
        from rich.console import Console

        self.console = console or Console()
        self.executor = executor or self._execute_with_main

    def run(self):
        self._banner()
        while True:
            action = self._choose_action()
            if action == 'q':
                self.console.print('[green]Goodbye.[/green]')
                return 0

            try:
                preview = self._collect_command(action)
            except KeyboardInterrupt:
                self.console.print('\n[yellow]Cancelled.[/yellow]')
                return 130
            except EOFError:
                self.console.print('\n[yellow]Input closed.[/yellow]')
                return 1

            if preview is None:
                continue

            self._show_preview(preview)
            if not self._confirm('Run this command now?', default=True):
                continue

            try:
                status = self.executor(preview.argv)
            except Exception as exc:
                self.console.print('[red]Command failed:[/red] {}'.format(exc))
                status = 1

            if status not in (None, 0):
                return status
            if not self._confirm('Run another pyBaram workflow?', default=False):
                return 0

    def _banner(self):
        from rich.panel import Panel

        self.console.print(Panel(
            '[bold]pyBaram Full TUI[/bold]\n'
            'Guided launcher for simulations, restarts, AOA sweeps, mesh '
            'import/partition, and export workflows.\n'
            'Simulation workflows default to the Rich live dashboard.',
            border_style='blue'
        ))

    def _choose_action(self):
        from rich.table import Table
        from rich.prompt import Prompt

        table = Table(title='Workflows', show_header=True)
        table.add_column('#', style='cyan', no_wrap=True)
        table.add_column('Workflow')
        table.add_column('What it does')
        rows = (
            ('1', 'Run simulation', 'Start a fresh mesh + config solve'),
            ('2', 'Restart simulation', 'Continue from a solution file'),
            ('3', 'AOA sweep', 'Run multiple angles of attack'),
            ('4', 'Import mesh', 'Convert an external mesh to .pbrm'),
            ('5', 'Partition mesh', 'Create a partitioned mesh'),
            ('6', 'Export solution', 'Write Tecplot/VTK/surface output'),
            ('q', 'Quit', 'Exit the launcher'),
        )
        for row in rows:
            table.add_row(*row)

        self.console.print(table)
        return Prompt.ask(
            'Choose workflow',
            choices=('1', '2', '3', '4', '5', '6', 'q'),
            default='1',
            console=self.console
        )

    def _collect_command(self, action):
        builders = {
            '1': self._collect_run,
            '2': self._collect_restart,
            '3': self._collect_sweep,
            '4': self._collect_import,
            '5': self._collect_partition,
            '6': self._collect_export,
            'q': lambda: None,
        }
        return builders[action]()

    def _collect_run(self):
        mesh = self._ask_existing_path('Mesh file (.pbrm)')
        ini = self._ask_existing_path('Config file (.ini)')
        ui = self._ask_ui(default='tui')
        return build_run_command(mesh, ini, ui)

    def _collect_restart(self):
        mesh = self._ask_existing_path('Mesh file (.pbrm)')
        soln = self._ask_existing_path('Solution file (.pbrs)')
        ini = self._ask_existing_path(
            'Override config file (.ini, blank to use solution config)',
            required=False
        )
        ui = self._ask_ui(default='tui')
        return build_restart_command(mesh, soln, ini or None, ui)

    def _collect_sweep(self):
        from rich.prompt import Prompt

        mesh = self._ask_existing_path('Mesh file (.pbrm)')
        ini = self._ask_existing_path('Base config file (.ini)')
        mode = Prompt.ask(
            'AOA input mode',
            choices=('values', 'range'),
            default='values',
            console=self.console
        )
        if mode == 'values':
            aoa_values = self._ask_nonempty(
                'AOA values, comma-separated',
                default='0,2,4'
            )
            aoa_range = None
        else:
            aoa_values = None
            aoa_range = (
                self._ask_nonempty('AOA start', default='0'),
                self._ask_nonempty('AOA stop', default='4'),
                self._ask_nonempty('AOA step', default='2'),
            )
        out = self._ask_output_path('Sweep output directory', default='sweep-aoa')
        ui = self._ask_ui(default='tui')
        existing = Prompt.ask(
            'Existing case directories',
            choices=('stop', 'overwrite', 'resume'),
            default='stop',
            console=self.console
        )
        return build_sweep_command(
            mesh,
            ini,
            aoa_values=aoa_values,
            aoa_range=aoa_range,
            out=out,
            ui=ui,
            overwrite=existing == 'overwrite',
            resume=existing == 'resume'
        )

    def _collect_import(self):
        inmesh = self._ask_existing_path('Input mesh')
        outmesh = self._ask_output_path('Output pyBaram mesh (.pbrm)')
        scale = self._ask_float('Scale', default='1')
        return build_import_command(inmesh, outmesh, scale)

    def _collect_partition(self):
        npart = self._ask_nonempty('Number of partitions', default='2')
        mesh = self._ask_existing_path('Mesh file (.pbrm)')
        soln_files = []
        while self._confirm('Add a solution file to partition with the mesh?',
                            default=False):
            soln_files.append(self._ask_existing_path('Solution file (.pbrs)'))
        out = self._ask_output_path('Output partitioned mesh')
        return build_partition_command(npart, mesh, out, tuple(soln_files))

    def _collect_export(self):
        mesh = self._ask_existing_path('Mesh file (.pbrm)')
        soln = self._ask_existing_path(
            'Solution file (.pbrs, blank if not needed)',
            required=False
        )
        out = self._ask_output_path(
            'Output file (blank to use command default)',
            required=False
        )
        surface = self._ask_nonempty(
            'Surface names, comma-separated (blank for volume export)',
            required=False
        )
        list_surfaces = self._confirm('List mesh surfaces before export?',
                                      default=False)
        return build_export_command(
            mesh,
            soln=soln or None,
            out=out or None,
            surface=surface or None,
            list_surfaces=list_surfaces
        )

    def _show_preview(self, preview):
        from rich.panel import Panel

        self.console.print(Panel(
            '[bold]{}[/bold]\n{}'
            .format(preview.title, preview.shell_command),
            title='Command preview',
            border_style='green'
        ))

    def _ask_existing_path(self, prompt, default=None, required=True):
        while True:
            value = self._ask_nonempty(prompt, default=default, required=required)
            if not value:
                return ''

            path = os.path.expanduser(value)
            if os.path.exists(path):
                return path

            self.console.print(
                "[red]Path does not exist:[/red] {}".format(path)
            )

    def _ask_output_path(self, prompt, default=None, required=True):
        value = self._ask_nonempty(prompt, default=default, required=required)
        return os.path.expanduser(value) if value else ''

    def _ask_nonempty(self, prompt, default=None, required=True):
        from rich.prompt import Prompt

        while True:
            value = Prompt.ask(
                prompt,
                default=default if default is not None else '',
                console=self.console
            )
            value = (value or '').strip()
            if value or not required:
                return value
            self.console.print('[red]A value is required.[/red]')

    def _ask_ui(self, default='tui'):
        from rich.prompt import Prompt

        return Prompt.ask(
            'Progress UI',
            choices=_UI_CHOICES,
            default=default,
            console=self.console
        )

    def _ask_float(self, prompt, default='1'):
        while True:
            value = self._ask_nonempty(prompt, default=default)
            try:
                return float(value)
            except ValueError:
                self.console.print('[red]Enter a number.[/red]')

    def _confirm(self, prompt, default=False):
        from rich.prompt import Confirm

        return Confirm.ask(prompt, default=default, console=self.console)

    @staticmethod
    def _execute_with_main(argv):
        from pybaram.__main__ import main

        return main(list(argv))


def _validate_ui(ui):
    if ui not in _UI_CHOICES:
        raise ValueError("Unknown progress UI {!r}".format(ui))
    return ui


def _format_number(value):
    return '{:.12g}'.format(float(value))
