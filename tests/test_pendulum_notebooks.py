"""Optional cells must be safe even if marimo schedules them independently."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import marimo as mo
from marimo._runtime.control_flow import MarimoStopError


class HiddenActorCriticTests(unittest.TestCase):
    def test_hidden_actor_cells_stop_before_reading_missing_training_variables(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ["overdamped_pendulum.py", "undamped_pendulum.py"]:
            with self.subTest(notebook=filename):
                path = root / filename
                spec = importlib.util.spec_from_file_location(path.stem, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.app._maybe_initialize()
                cells = [cell._cell for _, cell in module.app._cell_manager.valid_cells()]
                # Trace every descendant of the visibility toggle; do not rely
                # on a cell already having an explicit guard to include it.
                optional = {"show_actor_critic"}
                descendants = []
                pending = list(cells)
                while pending:
                    discovered = [cell for cell in pending if cell.refs & optional]
                    if not discovered:
                        break
                    for cell in discovered:
                        pending.remove(cell)
                        optional.update(cell.defs)
                        descendants.append(cell)
                self.assertGreaterEqual(len(descendants), 8)
                for cell in descendants:
                    with self.subTest(cell=cell.cell_id):
                        # None of the training/plot variables exist. A hidden
                        # cell must stop before touching any of them.
                        namespace = {
                            "mo": mo,
                            "show_actor_critic": SimpleNamespace(value=False),
                        }
                        with self.assertRaises(MarimoStopError):
                            exec(compile(cell.code, str(path), "exec"), namespace)


if __name__ == "__main__":
    unittest.main()
