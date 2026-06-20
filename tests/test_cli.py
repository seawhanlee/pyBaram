# -*- coding: utf-8 -*-
import unittest

from pybaram.__main__ import (
    build_parser,
    process_restart,
    process_run,
    process_sweep
)


class CliParserTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
