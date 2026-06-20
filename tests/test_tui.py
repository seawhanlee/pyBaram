# -*- coding: utf-8 -*-
import unittest

from pybaram.tui import (
    PyBaramTUI,
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
    def test_launcher_quits_from_menu(self):
        class QuitTUI(PyBaramTUI):
            def _banner(self):
                pass

            def _choose_action(self):
                return 'q'

        status = QuitTUI(console=FakeConsole()).run()

        self.assertEqual(status, 0)


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
