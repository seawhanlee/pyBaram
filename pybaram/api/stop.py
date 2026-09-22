"""Non-blocking keyboard cancellation shared by all MPI ranks."""
import os
import select
import sys
from contextlib import contextmanager


class SimulationStopped(Exception):
    """The user requested cancellation at an iteration boundary."""


class KeyboardStop:
    def __init__(self, comm):
        self.comm = comm
        self.fd = None

    def __call__(self):
        requested = False
        if self.fd is not None:
            try:
                if select.select([self.fd], [], [], 0)[0]:
                    data = os.read(self.fd, 4096)
                    requested = b'q' in data.lower()
                    if not data:
                        self.fd = None
            except (OSError, ValueError):
                self.fd = None
        if getattr(self.comm, 'size', 1) > 1:
            requested = self.comm.bcast(requested, root=0)
        if requested:
            raise SimulationStopped()

    @contextmanager
    def listening(self):
        settings = None
        fd = None
        if getattr(self.comm, 'rank', 0) == 0:
            try:
                fd = sys.stdin.fileno()
                if os.isatty(fd):
                    import termios
                    import tty
                    settings = termios.tcgetattr(fd)
                    tty.setcbreak(fd, termios.TCSANOW)
                self.fd = fd
            except (AttributeError, OSError, ValueError):
                self.fd = None
        try:
            yield self
        finally:
            self.fd = None
            if settings is not None:
                termios.tcsetattr(fd, termios.TCSANOW, settings)
