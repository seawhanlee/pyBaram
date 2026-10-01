"""Exercise the completion protocol without importing scientific dependencies."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pybaram.__main__ import build_parser


ROOT = Path(__file__).resolve().parents[1]
COMPLETE = '''
import sys
from importlib.abc import MetaPathFinder

class BlockSolverImports(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('pybaram.api', 'pybaram.readers',
                                'pybaram.solvers', 'pybaram.backends',
                                'numpy', 'mpi4py', 'h5py', 'rich')):
            raise AssertionError('Completion imported ' + fullname)

sys.meta_path.insert(0, BlockSolverImports())
from pybaram.__main__ import main
main()
raise AssertionError('Completion reached command dispatch')
'''


class CompletionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for name in ('mesh.pbrm', 'mesh.pbrmc', 'mesh.msh', 'mesh.cgns',
                     'solution.pbrs', 'second.pbrs', 'config.ini',
                     'config space.ini', 'other.txt', 'output.vtu'):
            (self.directory / name).touch()
        (self.directory / 'nested').mkdir()
        (self.directory / 'nested' / 'case.ini').touch()

    def complete(self, line, shell='bash'):
        result_file = self.directory / '.completions'
        env = dict(os.environ,
                   PYTHONPATH=os.pathsep.join(filter(None, (
                       str(ROOT), os.environ.get('PYTHONPATH', '')))),
                   COMP_LINE=line, COMP_POINT=str(len(line)),
                   _ARGCOMPLETE='1', _ARGCOMPLETE_SHELL=shell,
                   _ARGCOMPLETE_IFS='\v', _ARGCOMPLETE_SUPPRESS_SPACE='1',
                   _ARGCOMPLETE_STDOUT_FILENAME=str(result_file))
        before = {p.relative_to(self.directory): p.read_bytes()
                  for p in self.directory.rglob('*') if p.is_file()
                  and p != result_file}
        result = subprocess.run([sys.executable, '-c', COMPLETE],
                                cwd=self.directory, env=env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertEqual(result.stderr, '')
        after = {p.relative_to(self.directory): p.read_bytes()
                 for p in self.directory.rglob('*') if p.is_file()
                 and p != result_file}
        self.assertEqual(before, after)
        return set(filter(None, result_file.read_text().split('\v')))

    def test_commands_options_and_choices(self):
        self.assertEqual(self.complete('pybaram re'), {'restart'})
        commands = self.complete('pybaram ')
        self.assertTrue({'import', 'partition', 'run', 'restart', 'sweep',
                         'export', '--verbose', '-v'}.issubset(commands))
        for command in ('run', 'restart'):
            with self.subTest(command=command):
                self.assertEqual(self.complete(f'pybaram {command} --back'),
                                 {'--backend'})
                self.assertEqual(self.complete(f'pybaram {command} -b c'),
                                 {'cpu', 'cuda'})
                self.assertEqual(self.complete(f'pybaram {command} --backend=c'),
                                 {'--backend=cpu', '--backend=cuda'})
                self.assertEqual(self.complete(f'pybaram {command} --ui '),
                                 {'rich', 'none'})
        for command in ('import', 'partition'):
            self.assertEqual(self.complete(f'pybaram {command} -c '),
                             {'greedy', 'smallest-last'})

    def test_file_types_and_directories(self):
        self.assertEqual(self.complete('pybaram import mesh'),
                         {'mesh.msh', 'mesh.cgns', 'mesh.pbrm', 'mesh.pbrmc'})
        self.assertEqual(self.complete('pybaram run mesh'),
                         {'mesh.pbrm', 'mesh.pbrmc'})
        self.assertEqual(self.complete('pybaram run mesh.pbrm nested/'),
                         {'nested/case.ini'})
        self.assertEqual(self.complete('pybaram run mesh.pbrm missing'), set())
        self.assertEqual(self.complete('pybaram restart mesh.pbrm solution'),
                         {'solution.pbrs'})
        self.assertEqual(self.complete('pybaram restart mesh.pbrm solution.pbrs config.i'),
                         {'config.ini'})
        self.assertEqual(self.complete('pybaram sweep --out n'), {'nested/'})
        self.assertEqual(self.complete('pybaram sweep --out other'), set())

    def test_output_paths_are_unrestricted(self):
        for line in ('pybaram import mesh.msh output',
                     'pybaram export mesh.pbrm solution.pbrs output',
                     'pybaram partition 2 mesh.pbrm output',
                     'pybaram partition 2 mesh.pbrm solution.pbrs output'):
            self.assertEqual(self.complete(line), {'output.vtu'})

    def test_spaces_and_quotes(self):
        self.assertEqual(self.complete('pybaram run mesh.pbrm config\\ s'),
                         {'config\\ space.ini'})
        self.assertEqual(self.complete('pybaram run mesh.pbrm "config s'),
                         {'config space.ini'})

    def test_partition_variable_solutions(self):
        for line in ('pybaram partition 2 mesh.pbrm sol',
                     'pybaram partition 2 mesh.pbrm second.pbrs sol'):
            self.assertIn('solution.pbrs', self.complete(line))
        parsed = build_parser().parse_args([
            'partition', '2', 'mesh.pbrm',
            'solution.pbrs', 'second.pbrs', 'output.pbrm'
        ])
        self.assertEqual(parsed.soln, ['solution.pbrs', 'second.pbrs'])
        self.assertEqual(parsed.out, 'output.pbrm')

    def test_free_values_and_partial_sweeps(self):
        for line in ('pybaram partition other',
                     'pybaram import --scale other',
                     'pybaram export --surface other',
                     'pybaram sweep --aoa other',
                     'pybaram sweep --aoa-range -2 other'):
            with self.subTest(line=line):
                self.assertEqual(self.complete(line), set())
        for line in ('pybaram partition 2', 'pybaram import --scale 1',
                     'pybaram export --surface wall',
                     'pybaram sweep --aoa 0,',
                     'pybaram sweep --aoa-range -2 ',
                     'pybaram sweep --aoa-range -2 6 ',
                     'pybaram sweep --aoa-range -2 6 2 --ui n'):
            with self.subTest(line=line):
                completions = self.complete(line)
                self.assertNotIn('other.txt', completions)
        self.assertEqual(self.complete('pybaram sweep --aoa-range -2 6 2 --ui n'),
                         {'none'})
        for command in ('run', 'restart', 'sweep', 'export', 'import', 'partition'):
            self.complete(f'pybaram {command} ')

    def test_zsh_protocol(self):
        completions = self.complete('pybaram run --backend c', shell='zsh')
        self.assertEqual({item.split(':', 1)[0] for item in completions},
                         {'cpu', 'cuda'})


if __name__ == '__main__':
    unittest.main()
