# -*- coding: utf-8 -*-
import unittest
from unittest.mock import patch

from pybaram.__main__ import (
    build_parser,
    process_restart,
    process_run,
    process_sweep
)


class CliParserTest(unittest.TestCase):
    def test_backend_and_ui_are_forwarded_together(self):
        for command, files, process in (
            ('run', ['mesh.pbrm', 'conf.ini'], process_run),
            ('restart', ['mesh.pbrm', 'sol.pbrs', 'conf.ini'], process_restart),
        ):
            for backend in ('cpu', 'cuda'):
                with self.subTest(command=command, backend=backend):
                    args = build_parser().parse_args([
                        command, *files, '--backend', backend, '--ui', 'tui'
                    ])
                    with patch('pybaram.readers.native.NativeReader') as reader, \
                         patch('pybaram.inifile.INIFile') as config, \
                         patch('pybaram.api.simulation.' + command) as execute:
                        process(args)
                        expected = [reader.return_value]
                        if command == 'restart':
                            expected.append(reader.return_value)
                        expected.append(config.return_value)
                        execute.assert_called_once_with(*expected, be=backend, ui='tui')

    def test_cpu_remains_default_backend(self):
        for argv in (['run', 'mesh', 'config'], ['restart', 'mesh', 'solution']):
            self.assertEqual(build_parser().parse_args(argv).backend, 'cpu')

    def test_coloring_method_is_forwarded(self):
        for argv, function, positional in (
            (['import', 'mesh.msh', 'mesh.pbrmc'], 'import_mesh',
             ('mesh.msh', 'mesh.pbrmc', 1)),
            (['partition', '2', 'mesh.pbrm', 'mesh.pbrmc'], 'partition_mesh',
             ('mesh.pbrm', 'mesh.pbrmc', '2', [])),
        ):
            for method in ('greedy', 'smallest-last'):
                with self.subTest(command=argv[0], method=method):
                    args = build_parser().parse_args(argv + ['-c', method])
                    with patch('pybaram.api.io.' + function) as execute:
                        args.process(args)
                        execute.assert_called_once_with(*positional, coloring_method=method)

    def test_run_ui_defaults_to_tqdm(self):
        args = build_parser().parse_args(['run', 'mesh.pbrm', 'conf.ini'])

        self.assertEqual(args.ui, 'tqdm')
        self.assertIs(args.process, process_run)

    def test_run_accepts_tui(self):
        args = build_parser().parse_args([
            'run', 'mesh.pbrm', 'conf.ini', '--ui', 'tui'
        ])

        self.assertEqual(args.ui, 'tui')

    def test_restart_accepts_none(self):
        args = build_parser().parse_args([
            'restart', 'mesh.pbrm', 'sol.pbrs', '--ui', 'none'
        ])

        self.assertEqual(args.ui, 'none')
        self.assertIs(args.process, process_restart)

    def test_sweep_accepts_aoa_values(self):
        args = build_parser().parse_args([
            'sweep', 'mesh.pbrm', 'conf.ini', '--aoa', '0,2,4'
        ])

        self.assertEqual(args.aoa, '0,2,4')
        self.assertEqual(args.ui, 'tui')
        self.assertEqual(args.out, 'sweep-aoa')
        self.assertIs(args.process, process_sweep)

    def test_sweep_accepts_aoa_range(self):
        args = build_parser().parse_args([
            'sweep', 'mesh.pbrm', 'conf.ini',
            '--aoa-range', '0', '4', '2',
            '--ui', 'tqdm',
            '--out', 'runs',
            '--resume'
        ])

        self.assertEqual(args.aoa_range, ['0', '4', '2'])
        self.assertEqual(args.ui, 'tqdm')
        self.assertEqual(args.out, 'runs')
        self.assertTrue(args.resume)

    def test_sweep_rejects_resume_with_overwrite(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([
                'sweep', 'mesh.pbrm', 'conf.ini',
                '--aoa', '0',
                '--resume',
                '--overwrite'
            ])

    def test_tui_launcher_is_not_available(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(['tui'])


if __name__ == '__main__':
    unittest.main()
