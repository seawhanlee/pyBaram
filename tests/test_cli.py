# -*- coding: utf-8 -*-
import unittest

from pybaram.__main__ import build_parser, process_restart, process_run


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


if __name__ == '__main__':
    unittest.main()
