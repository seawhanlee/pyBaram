"""Keep autodoc2 0.5's inline class/method anchors compatible with Sphinx."""

from autodoc2.sphinx.autodoc import AutodocObject
from autodoc2.sphinx.utils import get_database


class InlineAutodocObject(AutodocObject):
    def run(self):
        full_name = self.arguments[0]
        item = get_database(self.env).get_item(full_name)
        original_context = self.env.ref_context

        class InlineContext(dict):
            def __setitem__(self, key, value):
                # autodoc2 sets a fully qualified class context before parsing
                # its short signatures. Sphinx expects a module-relative name,
                # and a top-level class must not be its own enclosing class.
                if key == 'py:class' and isinstance(value, str):
                    if item and item['type'] == 'class' and value == full_name:
                        value = None
                    else:
                        module = self.get('py:module')
                        if module and value.startswith(module + '.'):
                            value = value[len(module) + 1:]
                super().__setitem__(key, value)

        self.env.ref_context = InlineContext(original_context)
        try:
            return super().run()
        finally:
            self.env.ref_context = original_context


def setup(app):
    app.add_directive('autodoc2-object', InlineAutodocObject, override=True)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
