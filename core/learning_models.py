"""Model adapters for the shared learning lab.

All models expose current physical states, scalar actions, and a sparse matrix
whose row s*A+a gives the next-state distribution. The last column is a
zero-continuation terminal outcome, never an input to a reward function.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from . import pendulum_learning as pendulum
from . import walker_q_iteration as walker
from . import walker_stopping as stopping
from . import slip


MODEL_LABELS = {
    "Simple MDP": "mdp",
    "Overdamped pendulum · θ": "overdamped",
    "Inverted pendulum · θ, ω": "pendulum",
    "Inverted-pendulum walker (IPW) · section speed": "walker",
    "Spring-loaded inverted pendulum (SLIP) · apex height": "slip",
}

DEFAULTS = {
    "mdp": {"states": 7, "actions": 3, "concentration": 0.85, "seed": 4},
    "overdamped": {"angles": 81, "actions": 11, "torque_limit": 1.2, "dt": 0.08},
    "pendulum": {"angles": 61, "velocities": 31, "actions": 11,
                 "torque_limit": 1.2, "dt": 0.08, "damping": 0.03, "omega_limit": 3.0},
    "walker": {"states": 101, "actions": 33, "speed_limit": 4.43,
               "slope": 0.06, "alpha_min": float(np.pi / 8),
               "alpha_max": float(np.pi / 7)},
    "slip": slip.PAPER_PARAMS | {"states": 101, "actions": 81,
                                "alpha_min": float(np.deg2rad(-10)), "alpha_max": float(np.pi / 2)},
}

REWARDS = {
    "mdp": '''def reward(state, action, next_state, params):
    return 1.0 * (state == params["target_state"])
''',
    "overdamped": '''def reward(state, action, next_state, params):
    return np.cos(state) - params["torque_cost"] * action**2
''',
    "pendulum": '''def reward(state, action, next_state, params):
    theta, omega = state[..., 0], state[..., 1]
    return (np.cos(theta) - params["speed_cost"] * omega**2
            - params["torque_cost"] * action**2)
''',
    "walker": walker.DEFAULT_REWARD_SOURCE,
    "slip": '''def reward(state, action, next_state, params):
    # One reward per decision encourages continued hopping.
    # next_state is normalized next apex height, or NaN when no apex is reached.
    return 1.0
''',
}

INITIAL_SOURCE = '''def initial_q(state, action, params):
    # A scalar or an array broadcastable to (states, actions).
    # Try optimistic values: return 1.0 / (1.0 - params["discount"])
    return 0.0
'''


@dataclass
class LearningModel:
    kind: str
    states: np.ndarray  # S x dimension, including scalar models as S x 1
    actions: np.ndarray
    matrix: csr_matrix
    params: dict
    state_labels: tuple
    action_label: str
    axes: tuple
    context: dict = field(default_factory=dict)

    @property
    def shape(self):
        return len(self.states), len(self.actions)

    @property
    def terminal(self):
        return len(self.states)

    def continuation(self, values):
        return np.asarray(self.matrix @ np.r_[values, 0.0]).reshape(self.shape)

    def meshes(self):
        states = np.broadcast_to(self.states[:, None, :], self.shape + (self.states.shape[1],))
        if self.states.shape[1] == 1:
            states = states[..., 0]
        return states, np.broadcast_to(self.actions, self.shape)

def _count(settings, name, low=2, high=501):
    value = settings[name]
    if int(value) != value or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}.")
    return int(value)


def build_model(kind, settings=None):
    if kind not in DEFAULTS:
        raise ValueError(f"Unknown model: {kind}")
    p = DEFAULTS[kind] | (settings or {})
    if not all(np.isfinite(value) for value in p.values()):
        raise ValueError("Model parameters must be finite.")
    if kind == "walker":
        count, actions = _count(p, "states"), _count(p, "actions", high=129)
        if p["speed_limit"] <= 0 or p["alpha_min"] >= p["alpha_max"]:
            raise ValueError("Use a positive speed limit and alpha_min < alpha_max.")
        region = stopping.capture_region(float(p["slope"]))
        grid = walker.build_grid(np.linspace(0, p["speed_limit"], count),
                                 np.linspace(p["alpha_min"], p["alpha_max"], actions),
                                 walker.Walker(slope=p["slope"]), "linear", region=region)
        return LearningModel(kind, grid.states[:, None], grid.actions, grid.matrix,
                             grid.params | p, ("Section speed ω (rad/s)",), "Angle of attack α (rad)",
                             (grid.states,), {"grid": grid, "region": region,
                                              "next_states": walker.successor_speeds(grid)})
    if kind == "slip":
        count, action_count = _count(p, "states", high=301), _count(p, "actions", high=181)
        if not -np.pi <= p["alpha_min"] < p["alpha_max"] <= np.pi:
            raise ValueError("Use -π ≤ alpha_min < alpha_max ≤ π.")
        states = np.linspace(0, 1, count)
        actions = np.linspace(p["alpha_min"], p["alpha_max"], action_count)
        matrix, next_states, status = slip.build_grid(states, actions, p)
        return LearningModel(kind, states[:, None], actions, matrix, p,
                             ("Normalized apex height s = mgy/E",), "Angle of attack α (rad)",
                             (states,), {"next_states": next_states, "status": status})
    if kind == "mdp":
        count, actions = _count(p, "states", low=3, high=30), _count(p, "actions", high=8)
        if not 0 <= p["concentration"] <= 1 or int(p["seed"]) != p["seed"] or p["seed"] < 0:
            raise ValueError("Concentration must be in [0, 1] and seed a nonnegative integer.")
        rng = np.random.default_rng(int(p["seed"]))
        transitions = np.full((count, actions, count), (1 - p["concentration"]) / (count - 1))
        for action in range(actions):
            transitions[np.arange(count), action, np.arange(count)] = 0
            order = rng.permutation(count)
            transitions[order, action, np.roll(order, -1)] += p["concentration"]
        matrix = csr_matrix(np.pad(transitions.reshape(-1, count), ((0, 0), (0, 1))))
        states = np.arange(count, dtype=float)
        return LearningModel(kind, states[:, None], np.arange(actions), matrix,
                             p | {"target_state": count - 1}, ("State",), "Action", (states,))

    n_theta, n_actions = _count(p, "angles", high=241), _count(p, "actions", high=51)
    n_omega = _count(p, "velocities", low=3, high=161) if kind == "pendulum" else 1
    if p["torque_limit"] <= 0 or p["dt"] <= 0 or p.get("omega_limit", 1) <= 0:
        raise ValueError("Torque limit, timestep, and velocity limit must be positive.")
    plant = pendulum.Pendulum(kind="undamped" if kind == "pendulum" else "overdamped",
                             dt=p["dt"], torque_limit=p["torque_limit"],
                             damping=p.get("damping", 0), omega_limit=p.get("omega_limit", 3))
    theta = (np.arange(n_theta) - n_theta // 2) * (2 * np.pi / n_theta)
    omega = np.linspace(-plant.omega_limit, plant.omega_limit, n_omega)
    states = (theta[:, None] if n_omega == 1 else
              np.stack(np.meshgrid(theta, omega, indexing="ij"), axis=-1).reshape(-1, 2))
    actions = np.linspace(-plant.torque_limit, plant.torque_limit, n_actions)
    next_states = pendulum.step(plant, states[:, None, :], actions[None, :])
    ti = np.mod(next_states[..., 0] - theta[0], 2 * np.pi) * n_theta / (2 * np.pi)
    t0 = np.floor(ti).astype(int) % n_theta
    t1, wt = (t0 + 1) % n_theta, ti - np.floor(ti)
    columns, weights = [t0, t1], [1 - wt, wt]
    if n_omega > 1:
        wi = np.clip((next_states[..., 1] - omega[0]) / (omega[1] - omega[0]), 0, n_omega - 1)
        w0 = np.minimum(np.floor(wi).astype(int), n_omega - 2)
        ww = wi - w0
        outside = np.abs(next_states[..., 1]) > plant.omega_limit
        columns = [np.where(outside, len(states), t * n_omega + w)
                   for t, w in ((t0, w0), (t0, w0 + 1), (t1, w0), (t1, w0 + 1))]
        weights = [(1 - wt) * (1 - ww), (1 - wt) * ww, wt * (1 - ww), wt * ww]
    rows = np.arange(len(states) * n_actions)
    matrix = csr_matrix((np.concatenate([w.ravel() for w in weights]),
                         (np.tile(rows, len(weights)), np.concatenate([c.ravel() for c in columns]))),
                        shape=(len(rows), len(states) + 1))
    matrix.eliminate_zeros()
    labels = ("Angle θ (rad)",) if n_omega == 1 else ("Angle θ (rad)", "Velocity ω (rad/s)")
    return LearningModel(kind, states, actions, matrix,
                         p | {"torque_cost": 0.05, "speed_cost": 0.05}, labels, "Torque u",
                         (theta,) if n_omega == 1 else (theta, omega),
                         {"plant": plant, "next_states": next_states[..., 0] if n_omega == 1 else next_states})


def prepare_tables(model, reward_source, initial_source=INITIAL_SOURCE, *, params=None, discount=0.95):
    """Return expected rewards, initial Q, and optional CSR-edge rewards.

    Physical models use their actual successor before grid interpolation.
    Stochastic MDP rewards are evaluated separately at each possible successor;
    planning takes their expectation and RL uses the sampled edge's reward.
    A missing physical return is NaN, never an invented numeric state ID.
    """
    reward_params = model.params | (params or {}) | {"discount": discount}
    states, actions = model.meshes()
    initial = walker.evaluate_code(initial_source, "initial_q", (states, actions, reward_params),
                                   model.shape, discount=discount)
    edge_rewards = None
    if model.kind == "mdp":
        matrix = model.matrix
        rows = np.repeat(np.arange(np.prod(model.shape)), np.diff(matrix.indptr))
        successors = np.r_[model.states[:, 0], np.nan][matrix.indices]
        edge_rewards = walker.evaluate_code(reward_source, "reward",
            (states.ravel()[rows], actions.ravel()[rows], successors, reward_params),
            matrix.data.shape, discount=discount)
        reward = np.bincount(rows, weights=matrix.data * edge_rewards,
                             minlength=np.prod(model.shape)).reshape(model.shape)
    else:
        reward = walker.evaluate_code(reward_source, "reward",
            (states, actions, model.context["next_states"], reward_params), model.shape, discount=discount)
    return reward, initial, edge_rewards
