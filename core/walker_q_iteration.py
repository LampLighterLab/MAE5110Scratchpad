"""Event-driven walker return map, sparse Q iteration, and sampled Q-learning.

Conventions follow CodeAssignments/assignments/assignment_2.md and
CodeAssignments/instructor/inverted_pendulum_walker.py. The section is theta=0,
omega>0; alpha is half the inter-leg angle. The fixed bounded ankle-feedback
controller takes over on entering its empirical RoA, then maps to the absorbing
standing state omega=0. Rewards receive the current and next scalar speeds.
"""

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

from . import walker_stopping as stopping


DEFAULT_REWARD_SOURCE = '''def reward(state, action, next_state, params):
    # state = current section speed; action = alpha, both arrays of shape (S, A).
    # Capture maps to state=0, where the fixed ankle policy holds the walker.
    # Earn +1 per decision while standing, and 0 while walking.
    return 1.0 * (state == 0.0)
'''

DEFAULT_INITIAL_SOURCE = '''def initial_q(omega, alpha):
    # A scalar, state-only function, or full state-action function is allowed.
    # For optimistic initialization, try: return 1.0 / (1.0 - discount)
    # For V0(omega), try: return -omega**2
    # For action-dependent Q0, try: return 5.0 * np.cos(20.0 * alpha)
    return 0.0
'''


@dataclass(frozen=True)
class Walker:
    gravity: float = 9.81
    length: float = 1.0
    slope: float = 0.06

    def __post_init__(self):
        if not all(np.isfinite(x) for x in (self.gravity, self.length, self.slope)):
            raise ValueError("Physical parameters must be finite.")
        if self.gravity <= 0 or self.length <= 0:
            raise ValueError("Gravity and leg length must be positive.")
        if abs(self.slope) >= np.pi / 4:
            raise ValueError("The forward return map requires |slope| < pi/4.")


def return_map(omega, alpha, model=Walker()):
    """Section -> touchdown -> plastic impact -> next forward section crossing.

    Return NaN when there is no finite-time forward return. Require
    |slope| < alpha < pi/4: touchdown lies ahead of the section, the reset
    lies behind it, the hub stays above ground, and impact velocity is forward.
    The equality omega_next**2=0 is a separatrix, not a finite-time return.
    """
    omega, alpha = np.broadcast_arrays(
        np.asarray(omega, dtype=float), np.asarray(alpha, dtype=float)
    )
    if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(alpha)):
        raise ValueError("Speeds and actions must be finite.")
    if np.any((alpha <= abs(model.slope)) | (alpha >= np.pi / 4)):
        raise ValueError("Every action must satisfy |slope| < alpha < pi/4.")
    scale = 4 * model.gravity / model.length
    impact_speed_squared = omega**2 + scale * np.sin((model.slope + alpha) / 2)**2
    next_squared = (
        np.cos(2 * alpha)**2 * impact_speed_squared
        - scale * np.sin((model.slope - alpha) / 2)**2
    )
    return np.where(
        (omega > 0) & (next_squared > 0), np.sqrt(np.maximum(next_squared, 0)), np.nan
    )


@dataclass
class TransitionGrid:
    states: np.ndarray
    actions: np.ndarray
    next_speed: np.ndarray
    status: np.ndarray  # 0: return; 1: capture; 2: failure; 3: timeout; 4: grid exit
    source_goal: np.ndarray
    transition_states: np.ndarray
    impacts: np.ndarray
    matrix: csr_matrix  # row i*A+j; columns 0..S-1 and terminal S
    interpolation: str
    build_seconds: float
    params: dict

    @property
    def shape(self):
        return self.next_speed.shape

    @property
    def done(self):
        return self.status >= stopping.FAILURE

    @property
    def terminal(self):
        return len(self.states)

    def meshes(self):
        return np.meshgrid(self.states, self.actions, indexing="ij")

    def continuation(self, q):
        """P max_a Q; the extra terminal column always has value zero."""
        values = np.append(np.max(q, axis=1), 0.0)
        return np.asarray(self.matrix @ values).reshape(self.shape)


def _axis(values, name):
    values = np.array(values, dtype=float, copy=True)
    if (values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0)):
        raise ValueError(f"{name} must contain at least two finite, increasing points.")
    return values


def build_grid(states, actions, model=Walker(), interpolation="nearest", *, region=None):
    """Numerically grid section -> standing/failure/next section, then form P.

    Capture is checked throughout stance and after impact, using the actual
    assignment controller's RoA. Captures map to the standing node at speed 0;
    that node self-loops. Passive returns interpolate only among walking nodes,
    so rounding cannot invent a stop. Failures alone use the terminal column.
    Linear weights interpolate V; they do not represent physical process noise.
    """
    start = perf_counter()
    states, actions = _axis(states, "States"), _axis(actions, "Actions")
    if states[0] != 0:
        raise ValueError("The state grid must start at 0, the absorbing standing state.")
    if interpolation not in ("nearest", "linear"):
        raise ValueError("Interpolation must be nearest or linear.")
    # Validate geometry, independent of the return's existence.
    return_map(states[:, None], actions[None, :], model)
    if model.gravity != 9.81 or model.length != 1.0:
        raise ValueError("The bundled ankle-controller RoA uses g=9.81 and length=1.")
    params = stopping.controller_params(model.slope)
    if region is None:
        region = stopping.capture_region(model.slope)
    transitions = stopping.simulate_transitions(states[:, None], actions[None, :], region, params)
    status = transitions["status"].copy()
    endpoint_speed = transitions["states"][1]
    outside = (endpoint_speed < states[0]) | (endpoint_speed > states[-1])
    status[(status == stopping.RETURN) & outside] = 4
    returning = status == stopping.RETURN
    captured = status == stopping.CAPTURE
    next_speed = np.where(captured, 0.0, np.where(returning, endpoint_speed, np.nan))
    source_goal = stopping.capture_margin(region, np.array([np.zeros_like(states), states])) >= 0
    walking = np.flatnonzero(~source_goal)
    terminal, count = len(states), next_speed.size
    rows = np.arange(count)
    columns_low = np.full(count, terminal, dtype=int)
    columns_low[captured.ravel()] = 0
    columns_high = columns_low.copy()
    fraction = np.zeros(count)
    active_rows = np.flatnonzero(returning.ravel())
    if active_rows.size:
        if not walking.size:
            raise ValueError("Grid has no walking nodes; increase state resolution or range.")
        nodes = states[walking]
        values = endpoint_speed.ravel()[active_rows]
        upper = np.clip(np.searchsorted(nodes, values), 0, len(nodes) - 1)
        lower = np.maximum(0, upper - 1)
        if interpolation == "nearest":
            choose_lower = np.abs(values - nodes[lower]) <= np.abs(nodes[upper] - values)
            columns_low[active_rows] = walking[np.where(choose_lower, lower, upper)]
            columns_high = columns_low.copy()
        else:
            denominator = nodes[upper] - nodes[lower]
            weights = np.divide(values - nodes[lower], denominator,
                                out=np.zeros_like(values), where=denominator > 0)
            fraction[active_rows] = np.clip(weights, 0, 1)
            columns_low[active_rows], columns_high[active_rows] = walking[lower], walking[upper]
    matrix = csr_matrix(
        (np.concatenate([1 - fraction, fraction]),
         (np.concatenate([rows, rows]), np.concatenate([columns_low, columns_high]))),
        shape=(count, terminal + 1),
    )
    matrix.eliminate_zeros()
    return TransitionGrid(states, actions, next_speed, status, source_goal,
                          transitions["states"], transitions["impacts"], matrix,
                          interpolation, perf_counter() - start, params.copy())


class EditorError(ValueError):
    """A user-defined reward or initialization could not be evaluated."""


def evaluate_code(source, name, arguments, shape, *, discount):
    """Evaluate editable local Python; scalar/array outputs broadcast to (S,A).

    This has the same privileges as other notebook Python cells. Arguments
    are read-only array copies or independent parameter dictionaries, so an
    accidental assignment cannot corrupt the map or its model parameters.
    """
    namespace = {"np": np, "discount": discount}
    try:
        exec(compile(source, f"<{name} editor>", "exec"), namespace)
        function = namespace.get(name)
        if not callable(function):
            raise ValueError(f"Define a function named {name}.")
        copies = [deepcopy(arg) if isinstance(arg, dict) else np.array(arg, copy=True)
                  for arg in arguments]
        for arg in copies:
            if isinstance(arg, np.ndarray):
                arg.setflags(write=False)
        output = np.asarray(function(*copies), dtype=float)
        values = np.broadcast_to(output, shape).copy()
        if not np.all(np.isfinite(values)):
            raise ValueError("Return only finite numbers.")
        return values
    except Exception as error:
        raise EditorError(f"{name}: {type(error).__name__}: {error}") from error


def successor_speeds(grid):
    """Physical section speed, zero for capture, NaN for a missing return.

    A return outside the grid still has a physical successor speed.
    """
    return np.where(grid.status == 1, 0.0,
                    np.where((grid.status == 0) | (grid.status == 4),
                             grid.transition_states[1], np.nan))


def prepare_tables(grid, reward_source, initial_source, *, discount, params=None):
    """Evaluate r(current speed, alpha, next speed, params).

    params contains model/controller parameters plus discount; callers may add
    reward-specific parameters. The numerical state/action mesh is unmodified,
    including speed zero. Failure and capture flags are never passed to reward.
    """
    omega, alpha = grid.meshes()
    reward_params = grid.params | ({} if params is None else params) | {"discount": discount}
    kwargs = dict(shape=grid.shape, discount=discount)
    reward = evaluate_code(reward_source, "reward", (omega, alpha, successor_speeds(grid), reward_params), **kwargs)
    initial_q = evaluate_code(initial_source, "initial_q", (omega, alpha), **kwargs)
    return reward, initial_q


@dataclass
class IterationResult:
    grid: TransitionGrid
    reward: np.ndarray
    discount: float
    checkpoints: dict
    residuals: np.ndarray  # ||T Q_k - Q_k|| infinity, including k=0 and final k
    seconds: np.ndarray
    tolerance: float
    checkpoint_stride: int

    @property
    def method(self):
        return "value_iteration"

    @property
    def iteration_unit(self):
        return "sweep"

    @property
    def updates(self):
        return len(self.residuals) - 1

    @property
    def converged(self):
        return self.residuals[-1] <= self.tolerance

    @property
    def final_q(self):
        return self.checkpoints[self.updates]

    def backup(self, q):
        return self.reward + self.discount * self.grid.continuation(q)

    def q_at(self, iteration):
        """Recover any exact iterate using bounded-memory checkpoints."""
        if not isinstance(iteration, (int, np.integer)) or not 0 <= iteration <= self.updates:
            raise ValueError("Iteration is outside the recorded run.")
        nearest = max(k for k in self.checkpoints if k <= iteration)
        q = self.checkpoints[nearest].copy()
        for _ in range(nearest, iteration):
            q = self.backup(q)
        return q


def q_iteration(grid, reward, initial_q, *, discount=0.95, tolerance=1e-8,
                max_updates=1000, history_bytes=64 * 1024**2):
    """Synchronous Q_{k+1}=R+discount*P max_a Q_k, with residual diagnostics.

    Rewards apply to the current state/action, including standing and failure
    actions. Standing self-loops and can earn reward repeatedly. Only failure
    has zero continuation. No physical simulation occurs inside this loop.
    """
    reward, q = _learning_inputs(grid, reward, initial_q, discount, tolerance, max_updates)
    capacity = max(2, int(history_bytes) // q.nbytes)
    stride = max(1, int(np.ceil(max_updates / (capacity - 1))))
    checkpoints = {0: q.copy()}
    residuals, seconds = [], []
    start = perf_counter()
    for iteration in range(max_updates + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            next_q = reward + discount * grid.continuation(q)
            residual = float(np.max(np.abs(next_q - q)))
        if not np.isfinite(residual):
            raise ValueError("Q iteration overflowed; reduce reward or initialization magnitudes.")
        residuals.append(residual)
        seconds.append(perf_counter() - start)
        if residual <= tolerance or iteration == max_updates:
            checkpoints[iteration] = q.copy()
            break
        q = next_q
        if (iteration + 1) % stride == 0:
            checkpoints[iteration + 1] = q.copy()
    return IterationResult(grid, reward, discount, checkpoints, np.array(residuals),
                           np.array(seconds), tolerance, stride)


def _learning_inputs(grid, reward, initial_q, discount, tolerance, max_updates):
    if not np.isfinite(discount) or not 0 <= discount < 1:
        raise ValueError("Use 0 <= discount < 1 for a discounted contraction.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive.")
    if not isinstance(max_updates, (int, np.integer)) or max_updates < 0:
        raise ValueError("max_updates must be a nonnegative integer.")
    reward, q = np.array(reward, dtype=float, copy=True), np.array(initial_q, dtype=float, copy=True)
    if reward.shape != grid.shape or q.shape != grid.shape:
        raise ValueError("Reward and initial Q must have shape (states, actions).")
    if not np.all(np.isfinite(reward)) or not np.all(np.isfinite(q)):
        raise ValueError("Reward and initial Q must be finite.")
    return reward, q


@dataclass
class QLearningResult(IterationResult):
    sample_rows: np.ndarray
    sample_values: np.ndarray
    samples_per_iteration: int
    visit_counts: np.ndarray

    @property
    def method(self):
        return "q_learning"

    @property
    def iteration_unit(self):
        return "batch"

    def q_at(self, iteration):
        """Replay actual sampled updates, never substitute Bellman sweeps."""
        if not isinstance(iteration, (int, np.integer)) or not 0 <= iteration <= self.updates:
            raise ValueError("Iteration is outside the recorded run.")
        nearest = max(k for k in self.checkpoints if k <= iteration)
        q = self.checkpoints[nearest].copy()
        start, end = nearest * self.samples_per_iteration, iteration * self.samples_per_iteration
        flat = q.ravel()
        for row, value in zip(self.sample_rows[start:end], self.sample_values[start:end]):
            flat[row] = value
        return q


def q_learning(grid, reward, initial_q, *, discount=0.95, tolerance=1e-8,
               max_updates=1000, epsilon=0.3, learning_rate=0.2,
               samples_per_iteration=512, episode_length=32, seed=0,
               history_bytes=64 * 1024**2):
    """Tabular off-policy Q-learning along epsilon-greedy sampled trajectories.

    One displayed iteration is a batch of samples_per_iteration sequential TD
    updates. Episodes start uniformly over grid states and reset after failure
    or episode_length samples. Standing continues to self-loop and earn reward;
    a time-limit reset does not remove the bootstrap from the preceding update.
    Greedy behavior breaks ties randomly. Linear interpolation weights are
    sampled as transition probabilities in the gridded MDP, not physical noise.

    The full matrix is used only to report Bellman residuals, never to update Q.
    Checkpoints plus a compact (row, new value) update log give exact playback.
    history_bytes bounds checkpoint storage; the log uses 12 bytes per sample.
    A fixed learning rate is useful for experiments but need not converge under
    stochastic transitions. The residual, not elapsed batches, certifies accuracy.
    """
    reward, q = _learning_inputs(grid, reward, initial_q, discount, tolerance, max_updates)
    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0, 1].")
    if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must lie in (0, 1].")
    for name, value in (("samples_per_iteration", samples_per_iteration), ("episode_length", episode_length)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    n_states, n_actions = grid.shape
    capacity = max(2, int(history_bytes) // q.nbytes)
    stride = max(1, int(np.ceil(max_updates / (capacity - 1))))
    checkpoints = {0: q.copy()}
    sample_rows = np.empty(max_updates * samples_per_iteration, dtype=np.int32)
    sample_values = np.empty(max_updates * samples_per_iteration)
    visits = np.zeros(grid.shape, dtype=np.int64)
    flat_q, flat_reward, flat_visits = q.ravel(), reward.ravel(), visits.ravel()
    # Values are kept current after every individual TD update.
    state_values = q.max(axis=1)
    state, episode_step = int(rng.integers(n_states)), 0
    matrix = grid.matrix
    residuals, seconds = [], []
    start = perf_counter()
    count = 0
    for iteration in range(max_updates + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            residual = float(np.max(np.abs(reward + discount * grid.continuation(q) - q)))
        if not np.isfinite(residual):
            raise ValueError("Q-learning overflowed; reduce reward or initialization magnitudes.")
        residuals.append(residual)
        seconds.append(perf_counter() - start)
        if residual <= tolerance or iteration == max_updates:
            checkpoints[iteration] = q.copy()
            break
        for _ in range(samples_per_iteration):
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                ties = np.flatnonzero(q[state] == state_values[state])
                action = int(ties[rng.integers(len(ties))])
            row = state * n_actions + action
            lo, hi = matrix.indptr[row:row + 2]
            offset = 0 if hi - lo == 1 else int(np.searchsorted(
                np.cumsum(matrix.data[lo:hi]), rng.random(), side="right"))
            successor = int(matrix.indices[lo + min(offset, hi - lo - 1)])
            continuation = 0.0 if successor == grid.terminal else state_values[successor]
            target = flat_reward[row] + discount * continuation
            flat_q[row] += learning_rate * (target - flat_q[row])
            state_values[state] = q[state].max()
            flat_visits[row] += 1
            sample_rows[count], sample_values[count] = row, flat_q[row]
            count += 1
            episode_step += 1
            if successor == grid.terminal or episode_step >= episode_length:
                state, episode_step = int(rng.integers(n_states)), 0
            else:
                state = successor
        if (iteration + 1) % stride == 0:
            checkpoints[iteration + 1] = q.copy()
    return QLearningResult(grid, reward, discount, checkpoints, np.array(residuals),
                           np.array(seconds), tolerance, stride,
                           sample_rows[:count].copy(), sample_values[:count].copy(),
                           samples_per_iteration, visits)
