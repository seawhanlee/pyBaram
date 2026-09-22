import os
import pty
import termios
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pybaram.api.stop import KeyboardStop, SimulationStopped


class KeyboardStopTest(unittest.TestCase):
    def test_q_without_enter_and_terminal_restored(self):
        master, slave = pty.openpty()
        try:
            before = termios.tcgetattr(slave)
            stop = KeyboardStop(SimpleNamespace(rank=0, size=1))
            with patch('sys.stdin') as stdin:
                stdin.fileno.return_value = slave
                with self.assertRaises(SimulationStopped):
                    with stop.listening():
                        stop()  # No input must not block.
                        os.write(master, b'x')
                        stop()
                        os.write(master, b'q')
                        stop()
            self.assertEqual(termios.tcgetattr(slave), before)
            self.assertIsNone(stop.fd)
        finally:
            os.close(master)
            os.close(slave)

    def test_non_root_receives_cancellation_without_reading_stdin(self):
        comm = SimpleNamespace(rank=1, size=2, bcast=Mock(return_value=True))
        with patch('sys.stdin') as stdin:
            with KeyboardStop(comm).listening() as stop:
                with self.assertRaises(SimulationStopped):
                    stop()
            stdin.fileno.assert_not_called()
        comm.bcast.assert_called_once_with(False, root=0)

    def test_unavailable_stdin(self):
        with patch('sys.stdin') as stdin:
            stdin.fileno.side_effect = OSError('unavailable')
            with KeyboardStop(SimpleNamespace(rank=0, size=1)).listening() as stop:
                stop()
