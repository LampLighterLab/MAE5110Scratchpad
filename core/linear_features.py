"""One Python definition for each complete, linear-in-weights feature vector.

Suggested features only generate editable source. Training, coefficient plots,
and physical rollouts all evaluate that source with the run's copied parameters.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import inspect

import numpy as np


def suggestions(kind):
    """A small reference catalog: identifier -> label, Python expression, purpose."""
    bias = {"bias": ("1", "1.0", "A constant offset.")}
    if kind == "mdp":
        return bias | {
            "state": ("State ID", "state", "An affine baseline; numeric IDs impose an arbitrary ordering."),
            "target": ("At target", 'float(state == parameters["target_state"])', "Distinguish the rewarded state."),
            "one_hot": ("One-hot state vector", '*np.eye(int(parameters["states"]))[int(state)]', "One coefficient per state, equivalent to a table."),
        }
    if kind in ("pendulum", "overdamped"):
        result = bias | {
            "theta": ("θ", "theta", "Raw angle; useful near upright but discontinuous at the angle wrap."),
            "sin_theta": ("sin θ", "np.sin(theta)", "An odd, periodic feature distinguishing left from right."),
            "cos_theta": ("cos θ", "np.cos(theta)", "An even, periodic feature with a peak at upright."),
        }
        if kind == "pendulum":
            result |= {"omega": ("ω", "omega", "Distinguish the direction of angular motion."),
                       "omega_squared": ("ω²", "omega**2", "Represent a speed or kinetic-energy penalty.")}
        return result
    if kind == "walker":
        return bias | {
            "state": ("ω", "state", "An affine dependence on crossing speed."),
            "state_squared": ("ω²", "state**2", "Curvature in speed, proportional to kinetic energy."),
        }
    if kind == "slip":
        return bias | {
            "state": ("s", "state", "Normalized apex height."),
            "state_squared": ("s²", "state**2", "Curvature in apex height."),
            "forward_speed": ("Forward speed", 'np.sqrt(np.maximum(0.0, 2 * parameters["total_energy"] * (1 - state) / parameters["mass"]))',
                              "Physical apex speed at the model's fixed energy."),
        }
    raise ValueError(f"Unknown feature model: {kind}")


def default_names(kind):
    if kind == "pendulum":
        return ("bias", "theta", "omega")
    if kind == "overdamped":
        return ("bias", "theta")
    return ("bias", "state")


def source_for(kind, names=None):
    """Generate the full function; selections have no effect beyond this source."""
    names = default_names(kind) if names is None else tuple(names)
    catalog = suggestions(kind)
    if len(set(names)) != len(names) or set(names) - set(catalog):
        raise ValueError("Choose unique suggested features for this model.")
    lines = ["def features(state, parameters):"]
    if kind == "pendulum":
        lines.append("    theta, omega = state")
    elif kind == "overdamped":
        lines.append("    theta = state")
    if names:
        lines.append("    return np.array([")
        lines.extend(f"        {catalog[name][1]},  # {catalog[name][0]}" for name in names)
        lines.append("    ], dtype=float)")
    else:
        lines.append("    return np.array([], dtype=float)")
    return "\n".join(lines) + "\n"


class FeatureError(ValueError):
    """An editable feature function violates the state-to-vector contract."""


@dataclass
class FeatureBasis:
    """A run's complete function and copied parameters, shared by its snapshots."""

    model: object
    source: str
    label: str
    _parameters: dict = field(repr=False)
    _function: object = field(repr=False)
    _size: int | None = None
    grid: np.ndarray | None = field(default=None, repr=False)

    @property
    def labels(self):
        symbol = "ψ" if self.label == "Policy" else "φ"
        return tuple(f"{symbol}[{i}]" for i in range(self._size or 0))

    def _at(self, point):
        state = float(point[0]) if len(point) == 1 else np.array(point, copy=True)
        if isinstance(state, np.ndarray):
            state.setflags(write=False)
        try:
            output = self._function(state, deepcopy(self._parameters))
            if not isinstance(output, np.ndarray) or output.ndim != 1:
                raise ValueError("Return a one-dimensional NumPy array: np.array([feature_0, feature_1, ...]).")
            if np.iscomplexobj(output):
                raise ValueError("Return real feature values.")
            values = np.asarray(output, dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError("Return finite feature values for every state.")
            if self._size is not None and len(values) != self._size:
                raise ValueError(f"Return {self._size} features at every state; got {len(values)}.")
            return values.copy()
        except Exception as error:
            raise FeatureError(f"{self.label} features at state {state}: {type(error).__name__}: {error}") from error

    def evaluate(self, states=None):
        """Evaluate individual physical states; only the cached grid is used in TD updates."""
        points = self.model.states if states is None else np.asarray(states, dtype=float)
        if (points.ndim != 2 or points.shape[1] != self.model.states.shape[1]
                or not np.all(np.isfinite(points))):
            raise FeatureError(f"{self.label} features: states must be a finite (batch, state dimension) array.")
        if not len(points):
            return np.empty((0, self._size or 0))
        rows = []
        for point in points:
            row = self._at(point)
            if self._size is None:
                self._size = len(row)
            rows.append(row)
        return np.stack(rows)


def build_basis(model, source=None, parameters=None, *, label="Q"):
    """Compile features(state, parameters), returning the entire 1-D NumPy vector.

    Scalar models pass a scalar state; the pendulum passes [theta, omega].
    NumPy and ordinary local Python are available, as in the reward editor.
    There are no hidden or automatically appended columns.
    """
    source = source_for(model.kind) if source is None else source
    if not isinstance(source, str) or not source.strip():
        raise FeatureError(f"{label} features: define features(state, parameters).")
    try:
        namespace = {"np": np}
        exec(compile(source, f"<{label} features editor>", "exec"), namespace)
        function = namespace.get("features")
        if not callable(function):
            raise ValueError("Define features(state, parameters).")
        inspect.signature(function).bind(0.0, {})
    except Exception as error:
        raise FeatureError(f"{label} features: {type(error).__name__}: {error}") from error
    basis = FeatureBasis(model, source, label, deepcopy(model.params | (parameters or {})), function)
    grid = basis.evaluate()
    # Feature vectors must be repeatable: randomness or mutable global counters
    # would make training and deployment evaluate different functions.
    for i in np.unique(np.linspace(0, len(model.states) - 1, min(3, len(model.states))).astype(int)):
        if not np.allclose(basis._at(model.states[i]), grid[i], rtol=1e-10, atol=1e-12):
            raise FeatureError(f"{label} features: use a deterministic function of state and parameters.")
    grid.setflags(write=False)
    basis.grid = grid
    return basis
