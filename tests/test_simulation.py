import unittest
from unittest.mock import Mock, patch

from pybaram.api import simulation


class SimulationProgressTest(unittest.TestCase):
    def test_backend_selection_preserves_progress_and_cleanup(self):
        for backend_name in ('cpu', 'cuda'):
            for failing in (False, True):
                with self.subTest(backend=backend_name, failing=failing):
                    mesh, cfg, comm, context = (object() for _ in range(4))
                    backend, integrator, progress = Mock(), Mock(), Mock()
                    if failing:
                        integrator.run.side_effect = RuntimeError('solver failed')
                    with patch.object(simulation, 'get_backend', return_value=backend) as factory, \
                         patch.object(simulation, 'get_integrator', return_value=integrator) as get_intg, \
                         patch.object(simulation, 'add_progress_handler', return_value=progress) as handler:
                        def run():
                            simulation.run(mesh, cfg, be=backend_name, comm=comm,
                                           ui='tui', progress_context=context,
                                           suppress_final_status=True)
                        if failing:
                            with self.assertRaisesRegex(RuntimeError, 'solver failed'):
                                run()
                            progress.complete_context.assert_not_called()
                        else:
                            run()
                            progress.complete_context.assert_called_once_with(integrator)
                        factory.assert_called_once_with(backend_name, cfg, comm=comm)
                        get_intg.assert_called_once_with(backend, cfg, mesh, None, comm)
                        handler.assert_called_once_with(integrator, comm, 'tui', context)
                        self.assertTrue(integrator._suppress_final_status)
                        progress.start.assert_called_once_with()
                        progress.stop.assert_called_once_with()

    def test_restart_retains_backend_object_and_ui(self):
        mesh = {'mesh_uuid': 'same'}
        soln = {'mesh_uuid': 'same'}
        cfg, comm, backend = (object() for _ in range(3))
        with patch.object(simulation, 'get_backend') as factory, \
             patch.object(simulation, 'get_integrator') as get_intg, \
             patch.object(simulation, 'add_progress_handler') as handler:
            simulation.restart(mesh, soln, cfg, be=backend, comm=comm, ui='none')
            factory.assert_not_called()
            get_intg.assert_called_once_with(backend, cfg, mesh, soln, comm)
            handler.assert_called_once_with(get_intg.return_value, comm, 'none', None)
            handler.return_value.stop.assert_called_once_with()

    def test_unsupported_backend_is_rejected_before_progress(self):
        with patch.object(simulation, 'add_progress_handler') as handler:
            with self.assertRaisesRegex(ValueError, 'Unsupported backend'):
                simulation.run({}, {}, be='invalid', comm=object())
            handler.assert_not_called()
