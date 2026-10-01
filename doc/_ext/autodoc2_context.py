"""Keep autodoc2 0.5's inline class/method anchors compatible with Sphinx."""

from autodoc2.sphinx.autodoc import AutodocObject
from autodoc2.sphinx.utils import get_database
from autodoc2.render.myst_ import MystRenderer


class InlineMystRenderer(MystRenderer):
    """Render module APIs inside the guide's current section."""

    def render_module(self, item):
        full_name = item['full_name']
        yield f'```{{py:module}} {full_name}'
        if self.no_index(item):
            yield ':noindex:'
        yield '```'
        yield ''
        if self.show_docstring(item):
            yield f'```{{autodoc2-docstring}} {full_name}'
            if parser := self.get_doc_parser(full_name):
                yield f':parser: {parser}'
            yield '```'
            yield ''

        children = [child for child in self.get_children(item)
                    if child['type'] not in ('module', 'package')]
        if children and self.show_module_summary(item):
            yield from self.generate_summary(
                children,
                alias={child['full_name']: child['full_name'].split('.')[-1]
                       for child in children},
            )
            yield ''
        for child in children:
            yield from self.render_item(child['full_name'])


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
