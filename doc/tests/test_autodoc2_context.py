"""Regression coverage for inline autodoc2 API links (requires doc dependencies)."""

import io
from pathlib import Path
import tempfile
import unittest

from sphinx.application import Sphinx


class Autodoc2ContextTests(unittest.TestCase):
    def test_inline_class_and_method_keep_module_relative_anchors(self):
        extension_path = Path(__file__).resolve().parents[1] / '_ext'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'src'
            source.mkdir()
            (source / 'sample.py').write_text(
                'class Base:\n'
                '    def inherited(self):\n'
                '        """An inherited method."""\n'
                '        pass\n\n'
                'class Solver(Base):\n'
                '    """A sample solver."""\n'
                '    def step(self, dt):\n'
                '        """Advance one step."""\n'
                '        pass\n'
            )
            (source / 'conf.py').write_text(
                f'import sys\nsys.path.insert(0, {str(extension_path)!r})\n'
                "extensions = ['myst_parser', 'autodoc2', 'autodoc2_context']\n"
                "source_suffix = {'.md': 'markdown'}\n"
                "autodoc2_packages = [{'path': 'sample.py', 'auto_mode': False}]\n"
                "autodoc2_render_plugin = 'myst'\n"
                "autodoc2_hidden_objects = []\n"
                "autodoc2_class_inheritance = False\n"
                "master_doc = 'index'\n"
            )
            (source / 'index.md').write_text(
                '# API\n\n'
                '```{autodoc2-object} sample.Solver\n'
                'hidden_regexes = ["sample\\\\.Solver\\\\.step"]\n'
                '```\n\n'
                '```{autodoc2-object} sample.Solver.step\n'
                '```\n'
            )
            warnings = io.StringIO()
            app = Sphinx(
                str(source), str(source), str(root / 'html'),
                str(root / 'doctrees'), 'html',
                status=io.StringIO(), warning=warnings,
                freshenv=True, warningiserror=True,
                confoverrides={'nitpicky': True},
            )
            app.build(force_all=True)
            self.assertEqual(app.statuscode, 0, warnings.getvalue())
            html = (root / 'html/index.html').read_text()
            for anchor in ('sample.Solver', 'sample.Solver.inherited',
                           'sample.Solver.step'):
                self.assertIn(f'id="{anchor}"', html)
            self.assertNotIn('id="sample.sample.', html)
            self.assertNotIn('Solver.Solver', html)


if __name__ == '__main__':
    unittest.main()
