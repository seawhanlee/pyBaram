# -*- coding: utf-8 -*-
from pybaram.api.sweep import format_sweep_value


class SweepProgressContext:
    def __init__(self, aoas):
        self._cases = [
            {'aoa': format_sweep_value(aoa), 'residual': 'pending'}
            for aoa in aoas
        ]
        self.current = 'pending'
        self.completed = 0
        self.stop_requested = False
        self._index = None

    @property
    def total(self):
        return len(self._cases)

    @property
    def rows(self):
        return [
            (case['aoa'], case['residual'])
            for case in self._cases
        ]

    def start_case(self, aoa, index):
        self.current = format_sweep_value(aoa)
        self._index = index
        self._cases[index]['residual'] = 'running'

    def update_case(self, residual):
        if self._index is None or residual is None:
            return

        self._cases[self._index]['residual'] = residual

    def request_stop(self):
        self.stop_requested = True

    def complete_case(self, residual=None):
        if self._index is None:
            return

        self._cases[self._index]['residual'] = residual or 'complete'
        self.completed = min(self._index + 1, self.total)
        if self.completed == self.total:
            self.current = 'complete'
