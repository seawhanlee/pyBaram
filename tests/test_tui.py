# -*- coding: utf-8 -*-
import tempfile
import time
import unittest

from pathlib import Path
from unittest.mock import patch

import pybaram.tui as tui_module

from pybaram.tui import (
    CommandPreview,
    CommandRunState,
    ExecutionEvent,
    FileBrowserState,
    PyBaramTUI,
    TUICommandRunner,
    WorkflowState,
    stream_preview_events,
    build_export_command,
    build_import_command,
    build_partition_command,
    build_restart_command,
    build_run_command,
    build_sweep_command
)


class FakeConsole:
    def __init__(self):
        self.messages = []

    def print(self, *args, **kwargs):
        self.messages.append(args)


class TUILauncherTest(unittest.TestCase):
    def test_launcher_reports_missing_textual_cleanly(self):
        console = FakeConsole()

        with patch.object(
            tui_module,
            '_make_textual_app_class',
            side_effect=ImportError('no textual')
        ):
            status = PyBaramTUI(console=console).run()

        self.assertEqual(status, 1)
        self.assertTrue(console.messages)
        self.assertIn('requires the textual package', console.messages[0][0])

    def test_launcher_runs_textual_app_when_available(self):
        class FakeApp:
            def __init__(self, browser, workflow, runner):
                self.browser = browser
                self.workflow = workflow
                self.runner = runner

            def run(self):
                return 0

        with patch.object(tui_module, '_make_textual_app_class', return_value=FakeApp):
            status = PyBaramTUI(cwd='.').run()

        self.assertEqual(status, 0)


class FileBrowserStateTest(unittest.TestCase):
    def test_lists_filters_and_changes_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'case.ini').write_text('[config]\n')
            (root / 'mesh.pbrm').write_text('mesh')
            subdir = root / 'nested'
            subdir.mkdir()

            state = FileBrowserState(root)

            self.assertEqual(state.cwd, root.resolve())
            self.assertEqual(
                [entry.display_name for entry in state.entries],
                ['nested/', 'case.ini', 'mesh.pbrm']
            )

            state.set_filter('mesh')
            self.assertEqual(
                [entry.display_name for entry in state.entries],
                ['mesh.pbrm']
            )

            self.assertTrue(state.change_dir(subdir))
            self.assertEqual(state.cwd, subdir.resolve())
            self.assertTrue(state.parent())
            self.assertEqual(state.cwd, root.resolve())

    def test_rejects_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = FileBrowserState(tmp)

            self.assertFalse(state.change_dir('missing'))
            assert state.error is not None
            self.assertIn('Not a directory', state.error)


class WorkflowStateTest(unittest.TestCase):
    def test_assigns_selected_path_to_active_field_and_builds_run_preview(self):
        state = WorkflowState()
        state.select_workflow('run')

        self.assertTrue(state.assign_path_to_active_field('/tmp/mesh.pbrm'))
        state.next_field()
        self.assertTrue(state.assign_path_to_active_field('/tmp/config.ini'))

        preview = state.build_preview()
        assert preview is not None

        self.assertEqual(
            preview.argv,
            ('run', '/tmp/mesh.pbrm', '/tmp/config.ini', '--ui', 'tui')
        )

    def test_builds_partition_preview_from_comma_separated_solution_files(self):
        state = WorkflowState()
        state.select_workflow('partition')
        state.set_field('npart', '4')
        state.set_field('mesh', 'mesh.pbrm')
        state.set_field('soln', 'a.pbrs, b.pbrs')
        state.set_field('out', 'mesh-part.pbrm')

        preview = state.build_preview()
        assert preview is not None

        self.assertEqual(
            preview.argv,
            (
                'partition', '4', 'mesh.pbrm',
                'a.pbrs', 'b.pbrs', 'mesh-part.pbrm'
            )
        )

    def test_missing_required_fields_block_preview(self):
        state = WorkflowState()
        state.select_workflow('run')

        self.assertIsNone(state.build_preview())
        assert state.error is not None
        self.assertIn('Missing required fields', state.error)

    def test_edits_non_path_fields_and_output_paths_for_import_preview(self):
        state = WorkflowState()
        state.select_workflow('import')
        state.set_field('inmesh', 'mesh.msh')
        state.next_field()
        state.set_active_field('mesh.pbrm')
        state.next_field()
        state.set_active_field('2.5')

        preview = state.build_preview()
        assert preview is not None

        self.assertEqual(
            preview.argv,
            ('import', 'mesh.msh', 'mesh.pbrm', '--scale', '2.5')
        )

    def test_builds_sweep_preview_from_aoa_range(self):
        state = WorkflowState()
        state.select_workflow('sweep')
        state.set_field('mesh', 'mesh.pbrm')
        state.set_field('ini', 'config.ini')
        state.set_field('aoa_mode', 'range')
        state.set_field('aoa_range', '0, 6, 2')

        preview = state.build_preview()
        assert preview is not None

        self.assertEqual(
            preview.argv,
            (
                'sweep', 'mesh.pbrm', 'config.ini',
                '--aoa-range', '0', '6', '2',
                '--out', 'sweep-aoa',
                '--ui', 'tui'
            )
        )

    def test_rejects_incomplete_sweep_aoa_range(self):
        state = WorkflowState()
        state.select_workflow('sweep')
        state.set_field('mesh', 'mesh.pbrm')
        state.set_field('ini', 'config.ini')
        state.set_field('aoa_mode', 'range')
        state.set_field('aoa_range', '0, 6')

        self.assertIsNone(state.build_preview())
        assert state.error is not None
        self.assertIn('AOA range requires 3', state.error)

    def test_cycles_choice_and_bool_fields(self):
        state = WorkflowState()
        state.select_workflow('sweep')
        state.field_index = 2  # aoa_mode
        self.assertEqual(state.cycle_active_choice(), 'range')
        state.field_index = 6  # ui
        self.assertEqual(state.cycle_active_choice(), 'tqdm')
        self.assertEqual(state.cycle_active_choice(), 'none')
        state.field_index = 7  # existing_case
        self.assertEqual(state.cycle_active_choice(), 'overwrite')

        state.select_workflow('export')
        state.field_index = 4  # list_surfaces
        self.assertEqual(state.cycle_active_choice(), 'true')
        self.assertEqual(state.cycle_active_choice(), 'false')


class TUICommandRunnerTest(unittest.TestCase):
    def test_execution_argv_uses_unbuffered_python_module_and_preview_args(self):
        runner = TUICommandRunner(python_executable='python')
        preview = build_run_command('mesh.pbrm', 'config.ini', ui='tui')

        self.assertEqual(
            runner.execution_argv(preview),
            [
                'python', '-u', '-m', 'pybaram',
                'run', 'mesh.pbrm', 'config.ini', '--ui', 'tui'
            ]
        )
        self.assertEqual(
            runner.execution_argv(preview)[4:],
            list(preview.argv)
        )
        self.assertEqual(
            preview.shell_command,
            'pybaram run mesh.pbrm config.ini --ui tui'
        )

    def test_iter_events_streams_unbuffered_child_output_before_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / 'pybaram'
            pkg.mkdir()
            (pkg / '__init__.py').write_text('')
            (pkg / '__main__.py').write_text(
                "import time\n"
                "print('first')\n"
                "time.sleep(0.35)\n"
                "print('second')\n"
            )
            runner = TUICommandRunner()
            start = time.perf_counter()
            events = runner.iter_events(CommandPreview('fake', ()), cwd=tmp)

            first = next(events)

            self.assertEqual(first.text, 'first')
            self.assertLess(time.perf_counter() - start, 0.3)
            remaining = list(events)
            self.assertEqual([event.text for event in remaining], [
                'second', 'exit status 0'
            ])

    def test_iter_events_streams_partial_carriage_return_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / 'pybaram'
            pkg.mkdir()
            (pkg / '__init__.py').write_text('')
            (pkg / '__main__.py').write_text(
                "import sys, time\n"
                "sys.stdout.write('progress 1')\n"
                "sys.stdout.flush()\n"
                "time.sleep(0.35)\n"
                "sys.stdout.write('\\rprogress 2\\n')\n"
                "sys.stdout.flush()\n"
            )
            runner = TUICommandRunner()
            start = time.perf_counter()
            events = runner.iter_events(CommandPreview('fake', ()), cwd=tmp)

            first = next(events)

            self.assertEqual(first.text, 'progress 1')
            self.assertLess(time.perf_counter() - start, 0.3)
            remaining_text = ''.join(event.text for event in events)
            self.assertIn('progress 2', remaining_text)
            self.assertIn('exit status 0', remaining_text)

    def test_run_state_rejects_concurrent_start_and_records_status(self):
        state = CommandRunState()

        self.assertTrue(state.start())
        self.assertFalse(state.start())
        state.append(ExecutionEvent('stdout', 'hello'))
        state.append(ExecutionEvent('status', 'exit status 3', 3))

        self.assertFalse(state.active)
        self.assertEqual(state.returncode, 3)
        self.assertEqual(len(state.output), 2)

    def test_stream_preview_events_delivers_incrementally(self):
        class FakeRunner:
            def iter_events(self, preview, cwd=None):
                yield ExecutionEvent('stdout', 'first')
                yield ExecutionEvent('stdout', 'second')
                yield ExecutionEvent('status', 'exit status 0', 0)

        events = []

        stream_preview_events(
            FakeRunner(),
            CommandPreview('fake', ('run',)),
            '.',
            events.append
        )

        self.assertEqual([event.text for event in events], [
            'first', 'second', 'exit status 0'
        ])
        self.assertEqual(events[-1].returncode, 0)

    def test_stream_preview_events_reports_runner_failure(self):
        class BrokenRunner:
            def iter_events(self, preview, cwd=None):
                raise OSError('boom')

        state = CommandRunState()
        self.assertTrue(state.start())

        stream_preview_events(
            BrokenRunner(),
            CommandPreview('fake', ('run',)),
            '.',
            state.append
        )

        self.assertFalse(state.active)
        self.assertEqual(state.returncode, 1)
        self.assertIn('boom', state.output[-1].text)


class TUICommandBuilderTest(unittest.TestCase):
    def test_run_command_defaults_to_rich_tui(self):
        preview = build_run_command('mesh.pbrm', 'config.ini')

        self.assertEqual(
            preview.argv,
            ('run', 'mesh.pbrm', 'config.ini', '--ui', 'tui')
        )
        self.assertEqual(
            preview.shell_command,
            'pybaram run mesh.pbrm config.ini --ui tui'
        )

    def test_restart_command_omits_blank_config(self):
        preview = build_restart_command('mesh.pbrm', 'sol.pbrs', ui='none')

        self.assertEqual(
            preview.argv,
            ('restart', 'mesh.pbrm', 'sol.pbrs', '--ui', 'none')
        )

    def test_restart_command_includes_override_config(self):
        preview = build_restart_command(
            'mesh.pbrm', 'sol.pbrs', 'restart.ini', ui='tqdm'
        )

        self.assertEqual(
            preview.argv,
            (
                'restart', 'mesh.pbrm', 'sol.pbrs', 'restart.ini',
                '--ui', 'tqdm'
            )
        )

    def test_sweep_command_supports_explicit_values_and_resume(self):
        preview = build_sweep_command(
            'mesh.pbrm',
            'config.ini',
            aoa_values='0,2,4',
            out='runs',
            resume=True
        )

        self.assertEqual(
            preview.argv,
            (
                'sweep', 'mesh.pbrm', 'config.ini',
                '--aoa', '0,2,4',
                '--out', 'runs',
                '--ui', 'tui',
                '--resume'
            )
        )

    def test_sweep_command_supports_range_and_overwrite(self):
        preview = build_sweep_command(
            'mesh.pbrm',
            'config.ini',
            aoa_range=('0', '4', '2'),
            ui='tqdm',
            overwrite=True
        )

        self.assertEqual(
            preview.argv,
            (
                'sweep', 'mesh.pbrm', 'config.ini',
                '--aoa-range', '0', '4', '2',
                '--out', 'sweep-aoa',
                '--ui', 'tqdm',
                '--overwrite'
            )
        )

    def test_sweep_command_rejects_ambiguous_aoa_inputs(self):
        with self.assertRaises(ValueError):
            build_sweep_command(
                'mesh.pbrm',
                'config.ini',
                aoa_values='0',
                aoa_range=('0', '2', '1')
            )

    def test_sweep_command_rejects_overwrite_with_resume(self):
        with self.assertRaises(ValueError):
            build_sweep_command(
                'mesh.pbrm',
                'config.ini',
                aoa_values='0',
                overwrite=True,
                resume=True
            )

    def test_import_command_formats_scale(self):
        preview = build_import_command('mesh.msh', 'mesh.pbrm', 2.0)

        self.assertEqual(
            preview.argv,
            ('import', 'mesh.msh', 'mesh.pbrm', '--scale', '2')
        )

    def test_partition_command_preserves_optional_solution_files(self):
        preview = build_partition_command(
            4, 'mesh.pbrm', 'mesh-part.pbrm', ('a.pbrs', 'b.pbrs')
        )

        self.assertEqual(
            preview.argv,
            (
                'partition', '4', 'mesh.pbrm',
                'a.pbrs', 'b.pbrs',
                'mesh-part.pbrm'
            )
        )

    def test_export_command_supports_surface_listing(self):
        preview = build_export_command(
            'mesh.pbrm',
            soln='sol.pbrs',
            out='out.vtu',
            surface='wall,inlet',
            list_surfaces=True
        )

        self.assertEqual(
            preview.argv,
            (
                'export', 'mesh.pbrm', 'sol.pbrs', 'out.vtu',
                '--surface', 'wall,inlet',
                '--list-surfaces'
            )
        )

    def test_invalid_progress_ui_is_rejected(self):
        with self.assertRaises(ValueError):
            build_run_command('mesh.pbrm', 'config.ini', ui='bad')


if __name__ == '__main__':
    unittest.main()
