# -*- coding: utf-8 -*-
"""Full-screen terminal launcher for pyBaram workflows.

The solver progress dashboard lives in :mod:`pybaram.api.progress`.  This
module provides the front-door TUI: command builders, testable browser/workflow
state, and a lazily imported Textual full-screen app for filesystem-first
workflow assembly.
"""
import os
import shlex
import subprocess
import sys

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


_UI_CHOICES = ('tui', 'tqdm', 'none')
_PATH_FIELD_KINDS = {'existing_path', 'optional_existing_path', 'output_path'}


@dataclass(frozen=True)
class CommandPreview:
    """Command assembled by the interactive launcher."""

    title: str
    argv: Tuple[str, ...]

    @property
    def shell_command(self):
        return shlex.join(('pybaram',) + self.argv)


@dataclass(frozen=True)
class FileEntry:
    """A filesystem entry shown by the TUI file browser."""

    path: str
    name: str
    is_dir: bool

    @property
    def display_name(self):
        return self.name + ('/' if self.is_dir else '')


class FileBrowserState:
    """Testable current-directory, listing, and filter state."""

    def __init__(self, cwd=None):
        self.cwd = Path(cwd or os.getcwd()).expanduser().resolve()
        self.query = ''
        self.selected_index = 0
        self.error = None

    @property
    def entries(self):
        entries = []
        try:
            children = sorted(
                self.cwd.iterdir(),
                key=lambda path: (not path.is_dir(), path.name.lower())
            )
        except OSError as exc:
            self.error = str(exc)
            return entries

        query = self.query.lower()
        for child in children:
            if query and query not in child.name.lower():
                continue
            entries.append(FileEntry(
                str(child),
                child.name,
                child.is_dir()
            ))

        if self.selected_index >= len(entries):
            self.selected_index = max(len(entries) - 1, 0)
        return entries

    @property
    def selected_entry(self):
        entries = self.entries
        if not entries:
            return None
        return entries[self.selected_index]

    def set_filter(self, query):
        self.query = (query or '').strip()
        self.selected_index = 0

    def move_selection(self, offset):
        entries = self.entries
        if not entries:
            self.selected_index = 0
            return None
        self.selected_index = max(
            0,
            min(self.selected_index + offset, len(entries) - 1)
        )
        return self.selected_entry

    def change_dir(self, path):
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self.cwd / target
        target = target.resolve()
        if not target.is_dir():
            self.error = 'Not a directory: {}'.format(target)
            return False
        self.cwd = target
        self.selected_index = 0
        self.query = ''
        self.error = None
        return True

    def enter_selected(self):
        entry = self.selected_entry
        if entry is None:
            return None
        if entry.is_dir:
            self.change_dir(entry.path)
        return entry

    def parent(self):
        return self.change_dir(self.cwd.parent)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: str = 'text'
    default: str = ''
    required: bool = True


@dataclass(frozen=True)
class WorkflowSpec:
    key: str
    label: str
    fields: tuple


_WORKFLOW_SPECS = (
    WorkflowSpec('run', 'Run simulation', (
        FieldSpec('mesh', 'Mesh file (.pbrm)', 'existing_path'),
        FieldSpec('ini', 'Config file (.ini)', 'existing_path'),
        FieldSpec('ui', 'Progress UI', 'choice', 'tui'),
    )),
    WorkflowSpec('restart', 'Restart simulation', (
        FieldSpec('mesh', 'Mesh file (.pbrm)', 'existing_path'),
        FieldSpec('soln', 'Solution file (.pbrs)', 'existing_path'),
        FieldSpec('ini', 'Override config (.ini)', 'optional_existing_path', '', False),
        FieldSpec('ui', 'Progress UI', 'choice', 'tui'),
    )),
    WorkflowSpec('sweep', 'AOA sweep', (
        FieldSpec('mesh', 'Mesh file (.pbrm)', 'existing_path'),
        FieldSpec('ini', 'Base config file (.ini)', 'existing_path'),
        FieldSpec('aoa_mode', 'AOA input mode', 'choice', 'values'),
        FieldSpec('aoa_values', 'AOA values', 'text', '0,2,4'),
        FieldSpec('aoa_range', 'AOA range start,stop,step', 'text', '0,4,2'),
        FieldSpec('out', 'Output directory', 'output_path', 'sweep-aoa'),
        FieldSpec('ui', 'Progress UI', 'choice', 'tui'),
        FieldSpec('existing_case', 'Existing cases', 'choice', 'stop'),
    )),
    WorkflowSpec('import', 'Import mesh', (
        FieldSpec('inmesh', 'Input mesh', 'existing_path'),
        FieldSpec('outmesh', 'Output pyBaram mesh (.pbrm)', 'output_path'),
        FieldSpec('scale', 'Scale', 'text', '1'),
    )),
    WorkflowSpec('partition', 'Partition mesh', (
        FieldSpec('npart', 'Partitions', 'text', '2'),
        FieldSpec('mesh', 'Mesh file (.pbrm)', 'existing_path'),
        FieldSpec('soln', 'Solution files, comma-separated', 'optional_existing_path', '', False),
        FieldSpec('out', 'Output partitioned mesh', 'output_path'),
    )),
    WorkflowSpec('export', 'Export solution', (
        FieldSpec('mesh', 'Mesh file (.pbrm)', 'existing_path'),
        FieldSpec('soln', 'Solution file (.pbrs)', 'optional_existing_path', '', False),
        FieldSpec('out', 'Output file', 'output_path', '', False),
        FieldSpec('surface', 'Surface names', 'text', '', False),
        FieldSpec('list_surfaces', 'List surfaces', 'bool', 'false', False),
    )),
)


def workflow_specs():
    return _WORKFLOW_SPECS


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

    argv = [mesh, ini]
    argv.insert(0, 'sweep')
    if aoa_values:
        argv.extend(('--aoa', aoa_values))
    else:
        if aoa_range is None:
            raise ValueError('Provide exactly one of aoa_values or aoa_range')
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


def build_import_command(inmesh: str, outmesh: str,
                         scale: Union[float, int, str] = 1.0):
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


class WorkflowState:
    """Testable workflow selection, field values, and preview generation."""

    def __init__(self):
        self.workflow_index = 0
        self.field_index = 0
        self.values = {
            spec.key: {field.name: field.default for field in spec.fields}
            for spec in _WORKFLOW_SPECS
        }
        self.error = None

    @property
    def workflows(self):
        return _WORKFLOW_SPECS

    @property
    def spec(self):
        return self.workflows[self.workflow_index]

    @property
    def fields(self):
        return self.spec.fields

    @property
    def active_field(self):
        return self.fields[self.field_index]

    @property
    def active_values(self):
        return self.values[self.spec.key]

    def select_workflow(self, key_or_index):
        if isinstance(key_or_index, int):
            index = key_or_index
        else:
            keys = [spec.key for spec in self.workflows]
            index = keys.index(key_or_index)
        self.workflow_index = index % len(self.workflows)
        self.field_index = 0
        self.error = None

    def next_workflow(self, offset=1):
        self.select_workflow(self.workflow_index + offset)

    def next_field(self, offset=1):
        self.field_index = (self.field_index + offset) % len(self.fields)

    def set_field(self, name, value):
        self.active_values[name] = str(value or '')
        self.error = None

    def set_active_field(self, value):
        self.set_field(self.active_field.name, value)

    def active_field_value(self):
        return self.active_values.get(self.active_field.name, '')

    def cycle_active_choice(self):
        field = self.active_field
        if field.name == 'ui':
            return self._cycle_value(field.name, _UI_CHOICES)
        if field.name == 'existing_case':
            return self._cycle_value(field.name, ('stop', 'overwrite', 'resume'))
        if field.name == 'aoa_mode':
            return self._cycle_value(field.name, ('values', 'range'))
        if field.kind == 'bool':
            current = _truthy(self.active_values.get(field.name))
            self.set_field(field.name, 'false' if current else 'true')
            return self.active_values[field.name]
        self.error = '{} is not a choice field'.format(field.label)
        return None

    def _cycle_value(self, name, choices):
        current = self.active_values.get(name)
        try:
            index = choices.index(current)
        except ValueError:
            index = -1
        value = choices[(index + 1) % len(choices)]
        self.set_field(name, value)
        return value

    def assign_path_to_active_field(self, path):
        field = self.active_field
        if field.kind not in _PATH_FIELD_KINDS:
            self.error = '{} is not a path field'.format(field.label)
            return False
        self.set_field(field.name, path)
        return True

    def missing_required_fields(self):
        return [
            field.label
            for field in self.fields
            if field.required and not self.active_values.get(field.name)
        ]

    def build_preview(self):
        missing = self.missing_required_fields()
        if missing:
            self.error = 'Missing required fields: {}'.format(', '.join(missing))
            return None

        values = self.active_values
        key = self.spec.key
        try:
            if key == 'run':
                return build_run_command(values['mesh'], values['ini'], values['ui'])
            if key == 'restart':
                return build_restart_command(
                    values['mesh'], values['soln'], values.get('ini') or None,
                    values['ui']
                )
            if key == 'sweep':
                existing = values.get('existing_case', 'stop')
                if values.get('aoa_mode') == 'range':
                    return build_sweep_command(
                        values['mesh'], values['ini'],
                        aoa_range=_split_required_count(
                            values.get('aoa_range'), 3, 'AOA range'
                        ),
                        out=values.get('out') or 'sweep-aoa',
                        ui=values.get('ui') or 'tui',
                        overwrite=existing == 'overwrite',
                        resume=existing == 'resume'
                    )
                return build_sweep_command(
                    values['mesh'], values['ini'],
                    aoa_values=values.get('aoa_values') or '0,2,4',
                    out=values.get('out') or 'sweep-aoa',
                    ui=values.get('ui') or 'tui',
                    overwrite=existing == 'overwrite',
                    resume=existing == 'resume'
                )
            if key == 'import':
                return build_import_command(
                    values['inmesh'], values['outmesh'],
                    values.get('scale') or '1'
                )
            if key == 'partition':
                soln = _split_optional_list(values.get('soln'))
                return build_partition_command(
                    values['npart'], values['mesh'], values['out'], soln
                )
            if key == 'export':
                return build_export_command(
                    values['mesh'],
                    soln=values.get('soln') or None,
                    out=values.get('out') or None,
                    surface=values.get('surface') or None,
                    list_surfaces=_truthy(values.get('list_surfaces'))
                )
        except (TypeError, ValueError) as exc:
            self.error = str(exc)
            return None

        self.error = 'Unknown workflow {}'.format(key)
        return None


@dataclass
class ExecutionEvent:
    stream: str
    text: str
    returncode: Optional[int] = None


@dataclass
class CommandRunState:
    active: bool = False
    returncode: Optional[int] = None
    output: List[ExecutionEvent] = field(default_factory=list)

    def start(self):
        if self.active:
            return False
        self.active = True
        self.returncode = None
        self.output = []
        return True

    def append(self, event):
        self.output.append(event)
        if event.returncode is not None:
            self.returncode = event.returncode
            self.active = False


def stream_preview_events(runner, preview, cwd, on_event):
    """Stream runner events to a sink and always finish run state."""

    try:
        for event in runner.iter_events(preview, cwd=cwd):
            on_event(event)
    except Exception as exc:
        on_event(ExecutionEvent('stderr', 'command failed: {}'.format(exc), 1))


class TUICommandRunner:
    """Subprocess execution adapter for the Textual TUI."""

    def __init__(self, python_executable=None):
        self.python_executable = python_executable or sys.executable

    def execution_argv(self, preview):
        return [self.python_executable, '-u', '-m', 'pybaram'] + list(preview.argv)

    def iter_events(self, preview, cwd=None):
        proc = subprocess.Popen(
            self.execution_argv(preview),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0
        )
        try:
            for text in _iter_stream_chunks(proc.stdout):
                yield ExecutionEvent('stdout', text)
            returncode = proc.wait()
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.poll() is None:
                proc.terminate()
        yield ExecutionEvent('status', 'exit status {}'.format(returncode), returncode)


class PyBaramTUI:
    """Textual-powered interactive launcher for all pyBaram workflows."""

    def __init__(self, console=None, executor=None, cwd=None):
        self.console = console
        self.executor = executor or TUICommandRunner()
        self.cwd = cwd

    def run(self):
        try:
            app_cls = _make_textual_app_class()
        except ImportError as exc:
            message = (
                'pybaram tui requires the textual package. Install pyBaram '
                'with its runtime dependencies, or install textual separately. '
                'Missing import: {}'.format(exc)
            )
            if self.console is not None:
                self.console.print('[red]{}[/red]'.format(message))
            else:
                print(message, file=sys.stderr)
            return 1

        app = app_cls(
            browser=FileBrowserState(self.cwd),
            workflow=WorkflowState(),
            runner=self.executor
        )
        return app.run()


def _make_textual_app_class():
    from textual.app import App, ComposeResult  # type: ignore[reportMissingImports]
    from textual.containers import Horizontal, Vertical  # type: ignore[reportMissingImports]
    from textual.widgets import Footer, Header, Input, RichLog, Static  # type: ignore[reportMissingImports]

    class PyBaramTextualApp(App):
        """Runtime Textual application class built lazily for optional import."""

        CSS = """
        Screen { layout: vertical; }
        #main { height: 1fr; }
        #left { width: 45%; }
        #right { width: 55%; }
        .pane { border: solid $primary; padding: 0 1; }
        #files { height: 1fr; }
        #output { height: 1fr; }
        """
        BINDINGS = [
            ('q', 'quit', 'Quit'),
            ('tab', 'next_focus', 'Focus'),
            ('j', 'down', 'Down'),
            ('k', 'up', 'Up'),
            ('enter', 'open_or_assign', 'Open/select'),
            ('backspace', 'parent', 'Parent'),
            ('w', 'next_workflow', 'Workflow'),
            ('f', 'next_field', 'Field'),
            ('c', 'cycle_choice', 'Cycle choice'),
            ('r', 'run_command', 'Run'),
        ]

        def __init__(self, browser, workflow, runner):
            super().__init__()
            self.browser = browser
            self.workflow = workflow
            self.runner = runner
            self.run_state = CommandRunState()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id='main'):
                with Vertical(id='left'):
                    yield Static(id='cwd', classes='pane')
                    yield Input(placeholder='Filter files / autocomplete paths', id='filter')
                    yield Input(placeholder='Edit active field and press Enter', id='field-input')
                    yield Static(id='files', classes='pane')
                with Vertical(id='right'):
                    yield Static(id='workflow', classes='pane')
                    yield Static(id='fields', classes='pane')
                    yield Static(id='preview', classes='pane')
                    yield RichLog(id='output', classes='pane', wrap=True)
            yield Footer()

        def on_mount(self):
            self._refresh_all()
            self.query_one('#filter', Input).focus()

        def on_input_changed(self, event):
            if event.input.id == 'filter':
                self.browser.set_filter(event.value)
                self._refresh_browser()

        def on_input_submitted(self, event):
            if event.input.id == 'field-input':
                self.workflow.set_active_field(event.value)
                self.workflow.next_field()
                self._sync_field_input()
                self._refresh_workflow()

        def action_next_focus(self):
            self.screen.focus_next()

        def action_down(self):
            self.browser.move_selection(1)
            self._refresh_browser()

        def action_up(self):
            self.browser.move_selection(-1)
            self._refresh_browser()

        def action_parent(self):
            self.browser.parent()
            self._refresh_all()

        def action_next_workflow(self):
            self.workflow.next_workflow()
            self._refresh_workflow()

        def action_next_field(self):
            self.workflow.next_field()
            self._sync_field_input()
            self._refresh_workflow()

        def action_cycle_choice(self):
            self.workflow.cycle_active_choice()
            self._sync_field_input()
            self._refresh_workflow()

        def action_open_or_assign(self):
            entry = self.browser.selected_entry
            if entry is None:
                return
            if entry.is_dir:
                self.browser.change_dir(entry.path)
                self.query_one('#filter', Input).value = ''
                self._refresh_all()
                return
            if self.workflow.assign_path_to_active_field(entry.path):
                self.workflow.next_field()
                self._sync_field_input()
            self._refresh_workflow()

        def action_run_command(self):
            preview = self.workflow.build_preview()
            self._refresh_preview(preview)
            if preview is None:
                return
            if not self.run_state.start():
                self._log('A command is already running.')
                return
            self._log('$ {}'.format(preview.shell_command))
            self.run_worker(self._run_preview(preview), exclusive=True)

        async def _run_preview(self, preview):
            import asyncio

            loop = asyncio.get_running_loop()

            def on_event(event):
                loop.call_soon_threadsafe(self._handle_execution_event, event, preview)

            await asyncio.to_thread(
                stream_preview_events,
                self.runner,
                preview,
                str(self.browser.cwd),
                on_event
            )

        def _handle_execution_event(self, event, preview):
            self.run_state.append(event)
            self._log(event.text)
            self._refresh_preview(preview)

        def _refresh_all(self):
            self._refresh_browser()
            self._refresh_workflow()

        def _refresh_browser(self):
            self.query_one('#cwd', Static).update(
                '[bold]Current directory[/bold]\n{}\n[dim]Filter: {}[/dim]'
                .format(self.browser.cwd, self.browser.query or '(none)')
            )
            rows = []
            entries = self.browser.entries
            if not entries:
                rows.append('[dim](no matches)[/dim]')
            for index, entry in enumerate(entries[:200]):
                marker = '>' if index == self.browser.selected_index else ' '
                style = 'cyan' if entry.is_dir else 'white'
                rows.append('{} [{}]{}[/{}]'.format(
                    marker, style, entry.display_name, style
                ))
            if self.browser.error:
                rows.append('[red]{}[/red]'.format(self.browser.error))
            self.query_one('#files', Static).update('\n'.join(rows))

        def _sync_field_input(self):
            field_input = self.query_one('#field-input', Input)
            field_input.placeholder = 'Edit {} and press Enter'.format(
                self.workflow.active_field.label
            )
            field_input.value = self.workflow.active_field_value()

        def _refresh_workflow(self):
            workflow_rows = []
            for index, spec in enumerate(self.workflow.workflows):
                marker = '>' if index == self.workflow.workflow_index else ' '
                workflow_rows.append('{} {}'.format(marker, spec.label))
            self.query_one('#workflow', Static).update(
                '[bold]Workflows[/bold]\n' + '\n'.join(workflow_rows)
            )

            field_rows = []
            values = self.workflow.active_values
            for index, field in enumerate(self.workflow.fields):
                marker = '>' if index == self.workflow.field_index else ' '
                value = values.get(field.name, '') or '[dim](blank)[/dim]'
                field_rows.append('{} {}: {}'.format(marker, field.label, value))
            self.query_one('#fields', Static).update(
                '[bold]Fields[/bold]\n' + '\n'.join(field_rows) +
                '\n[dim]Enter assigns selected file; type field value below and press Enter; c cycles choices/bools.[/dim]'
            )
            self._sync_field_input()
            self._refresh_preview(self.workflow.build_preview())

        def _refresh_preview(self, preview):
            if preview is None:
                content = self.workflow.error or 'Fill required fields to preview.'
            else:
                content = preview.shell_command
                if self.run_state.active:
                    content += '\n[yellow]Running...[/yellow]'
                elif self.run_state.returncode is not None:
                    content += '\nexit status {}'.format(self.run_state.returncode)
            self.query_one('#preview', Static).update(
                '[bold]Command preview[/bold]\n{}'.format(content)
            )

        def _log(self, line):
            self.query_one('#output', RichLog).write(line)

    return PyBaramTextualApp


def _iter_stream_chunks(stream, chunk_size=4096):
    """Yield decoded output chunks as soon as the child writes them.

    Solver progress UIs commonly update with carriage returns rather than
    newline-delimited log records, so line iteration would make the output pane
    look frozen until the next newline or process exit.  This reader forwards
    available bytes promptly and leaves presentation decisions to the TUI.
    """

    if stream is None:
        return

    fd = stream.fileno()
    decoder = None
    try:
        import codecs
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        while True:
            chunk = os.read(fd, chunk_size)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                yield text.rstrip('\n')
        tail = decoder.decode(b'', final=True)
        if tail:
            yield tail.rstrip('\n')
    except OSError:
        return


def _split_optional_list(value):
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(',') if item.strip())


def _split_required_count(value, count, label):
    items = _split_optional_list(value)
    if len(items) != count:
        raise ValueError('{} requires {} comma-separated values'.format(label, count))
    return items


def _truthy(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def _validate_ui(ui):
    if ui not in _UI_CHOICES:
        raise ValueError("Unknown progress UI {!r}".format(ui))
    return ui


def _format_number(value):
    return '{:.12g}'.format(float(value))
