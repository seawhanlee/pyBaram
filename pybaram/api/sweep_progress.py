# -*- coding: utf-8 -*-
from pybaram.api.sweep import format_sweep_value


class SweepProgressContext:
    def __init__(self, aoas):
        self._aoas = list(aoas)
        self.current = 'pending'
        self.completed = 0
        self._index = None

    @property
    def total(self):
        return len(self._aoas)

    def start_case(self, aoa, index):
        self.current = format_sweep_value(aoa)
        self._index = index

    def complete_case(self):
        if self._index is None:
            return

        self.completed = min(self._index + 1, self.total)
        if self.completed == self.total:
            self.current = 'complete'
