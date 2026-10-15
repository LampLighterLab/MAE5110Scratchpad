"""Comparable planning and sampled-learning experiments on a shared model.

The value representation and policy representation are independent choices.
Policies either maximize Q, fit a parameterized classifier to those actions,
or alternate policy evaluation and greedy improvement. Linear critics use
least-squares fits for full backups and normalized semi-gradient TD for samples.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

from . import linear_features


@dataclass(frozen=True)
class Settings:
    method: str = "planning"
    approximation: bool = False
    policy: str = "greedy"
    discount: float = 0.95
    iterations: int = 200
    learning_rate: float = 0.15
    epsilon: float = 0.3
    samples: int = 256
    episode_length: int = 32
    feature_source: str | None = None
    policy_feature_source: str | None = None
    evaluation_sweeps: int = 20
    seed: int = 0
    ridge: float = 1e-5

    def __post_init__(self):
        if self.method not in ("planning", "rl"):
            raise ValueError("Method must be planning or rl.")
        if self.policy not in ("greedy", "fit", "iteration"):
            raise ValueError("Policy must be greedy, fit, or iteration.")
        for name, lower, upper in (("discount", 0, 0.999), ("epsilon", 0, 1),
                                    ("learning_rate", 1e-6, 1),
                                    ("ridge", 1e-12, 1)):
            value = getattr(self, name)
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be in [{lower}, {upper}].")
        for name, lower, upper in (("iterations", 0, 2000), ("samples", 1, 4096),
                                    ("episode_length", 1, 10000), ("evaluation_sweeps", 1, 500),
                                    ("seed", 0, 2**32 - 1)):
            value = getattr(self, name)
            if int(value) != value or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}].")

    @property
    def label(self):
        if self.policy == "iteration":
            if self.method == "rl":
                return "Sampled policy iteration"
            return "Approximate policy iteration" if self.approximation else "Classical policy iteration"
        label = "Q-value iteration" if self.method == "planning" else "ε-greedy Q-learning"
        return label + (" + fitted policy" if self.policy == "fit" else "")


@dataclass
class Snapshot:
    iteration: int
    q: np.ndarray
    policy: np.ndarray
    visits: np.ndarray
    q_weights: np.ndarray | None = None
    policy_weights: np.ndarray | None = None
    feature_basis: linear_features.FeatureBasis | None = None
    policy_feature_basis: linear_features.FeatureBasis | None = None

    @property
    def value(self):
        return self.q.max(axis=1)


@dataclass
class Experiment:
    settings: Settings
    reward: np.ndarray
    snapshots: list
    residuals: np.ndarray
    changes: np.ndarray
    seconds: np.ndarray
    critic_parameters: int
    policy_parameters: int

    @property
    def final(self):
        return self.snapshots[-1]


def _greedy(q):
    policy = np.zeros_like(q)
    policy[np.arange(len(q)), np.argmax(q, axis=1)] = 1
    return policy


def _least_squares_projection(phi, ridge):
    """Ridge least squares, including dependent and empty feature vectors."""
    u, singular, vt = np.linalg.svd(phi, full_matrices=False)
    return (vt.T * (singular / (singular**2 + ridge))) @ u.T


class _Critic:
    def __init__(self, initial, settings, basis):
        self.features = basis.grid if basis is not None else None
        self.projection = None
        if self.features is not None:
            # Signed polynomial/trigonometric features require a least-squares
            # fit. Projection need not preserve bounds or Bellman contraction.
            self.projection = _least_squares_projection(self.features, settings.ridge)
        self.fit(initial)

    def fit(self, targets):
        self.weights = targets.copy() if self.features is None else self.projection @ targets

    def q(self, state=None):
        if self.features is None:
            return self.weights.copy() if state is None else self.weights[state]
        phi = self.features if state is None else self.features[state]
        return phi @ self.weights

    def update(self, state, action, error, rate):
        if self.features is None:
            self.weights[state, action] += rate * error
        else:
            phi = self.features[state]
            self.weights[:, action] += rate * error * phi / max(float(phi @ phi), 1e-12)


def train(model, reward, initial_q, settings=Settings(), *, transition_rewards=None, feature_parameters=None):
    """Train once; retain bounded snapshots for linked plots and rollouts.

    RL updates use only sampled transitions. Full-matrix residuals are read-only
    diagnostics. A batch is not a sweep: the GUI reports transition counts too.
    The time-limit reset keeps the preceding bootstrap; true failures do not.
    """
    reward, initial_q = np.asarray(reward, dtype=float), np.asarray(initial_q, dtype=float)
    if reward.shape != model.shape or initial_q.shape != model.shape:
        raise ValueError("Reward and initial Q must match the state-action grid.")
    if not np.all(np.isfinite(reward)) or not np.all(np.isfinite(initial_q)):
        raise ValueError("Reward and initial Q must be finite.")
    if transition_rewards is not None:
        transition_rewards = np.asarray(transition_rewards, dtype=float)
        if transition_rewards.shape != model.matrix.data.shape or not np.all(np.isfinite(transition_rewards)):
            raise ValueError("Transition rewards must be finite and aligned with the sparse matrix entries.")
    parameters = model.params | (feature_parameters or {}) | {"discount": settings.discount}
    q_basis = (linear_features.build_basis(model, settings.feature_source, parameters, label="Q")
               if settings.approximation else None)
    policy_basis = (linear_features.build_basis(model, settings.policy_feature_source,
                                                parameters, label="Policy") if settings.policy == "fit" else None)
    critic = _Critic(initial_q, settings, q_basis)
    policy_phi = policy_basis.grid if policy_basis is not None else None
    policy_projection = None
    if policy_phi is not None:
        policy_projection = _least_squares_projection(policy_phi, settings.ridge)
    policy = _greedy(critic.q())
    rng = np.random.default_rng(settings.seed)
    visits = np.zeros(model.shape, dtype=np.int64)
    state, episode_step = int(rng.integers(model.terminal)), 0
    # At most 81 snapshots and about 64 MiB; exact selected snapshots are shown.
    capacity = max(2, min(81, (64 * 1024**2) // (3 * initial_q.nbytes)))
    stride = max(1, int(np.ceil(settings.iterations / (capacity - 1))))
    snapshots, residuals, changes, seconds = [], [], [], []
    previous_q, started = None, perf_counter()
    for iteration in range(settings.iterations + 1):
        q = critic.q()
        if not np.all(np.isfinite(q)) or np.max(np.abs(q)) > 1e12:
            raise ValueError("Learning diverged. Try a different feature vector or stronger regularization for planning; "
                             "for RL, also try a smaller learning rate. Check the initialization and reward scale.")
        if settings.policy == "fit":
            # Multiclass least-squares classifier trained on greedy action
            # labels, not policy gradients or supervised Q targets.
            policy_weights = policy_projection @ _greedy(q)
            policy = _greedy(policy_phi @ policy_weights)
        elif settings.policy == "greedy":
            policy = _greedy(q)
        value = (np.sum(policy * q, axis=1) if settings.policy == "iteration" else q.max(axis=1))
        target = reward + settings.discount * model.continuation(value)
        residuals.append(float(np.max(np.abs(target - q))))
        changes.append(0.0 if previous_q is None else float(np.max(np.abs(q - previous_q))))
        seconds.append(perf_counter() - started)
        if iteration % stride == 0 or iteration == settings.iterations:
            snapshots.append(Snapshot(iteration, q.copy(), policy.copy(), visits.copy(),
                                      critic.weights.copy() if settings.approximation else None,
                                      policy_weights.copy() if settings.policy == "fit" else None,
                                      q_basis, policy_basis))
        if iteration == settings.iterations:
            break
        previous_q = q
        if settings.method == "planning":
            if settings.policy == "iteration" and not settings.approximation:
                evaluated = evaluate_policy(model, reward, policy, settings.discount)
                critic.fit(reward + settings.discount * model.continuation(evaluated))
            elif settings.policy == "iteration":
                for _ in range(settings.evaluation_sweeps):
                    evaluated = np.sum(policy * critic.q(), axis=1)
                    critic.fit(reward + settings.discount * model.continuation(evaluated))
            else:
                critic.fit(target)
        else:
            for _ in range(settings.samples):
                qs = critic.q(state)
                if rng.random() < settings.epsilon:
                    action = int(rng.integers(model.shape[1]))
                elif settings.policy == "iteration":
                    action = int(np.argmax(policy[state]))
                else:
                    ties = np.flatnonzero(qs == qs.max())
                    action = int(rng.choice(ties))
                row = state * model.shape[1] + action
                lo, hi = model.matrix.indptr[row:row + 2]
                edge = int(rng.choice(np.arange(lo, hi), p=model.matrix.data[lo:hi]))
                successor = int(model.matrix.indices[edge])
                sampled_reward = reward[state, action] if transition_rewards is None else transition_rewards[edge]
                continuation = 0.0
                if successor != model.terminal:
                    following = critic.q(successor)
                    continuation = (float(policy[successor] @ following) if settings.policy == "iteration"
                                    else float(following.max()))
                error = sampled_reward + settings.discount * continuation - qs[action]
                critic.update(state, action, error, settings.learning_rate)
                visits[state, action] += 1
                episode_step += 1
                if successor == model.terminal or episode_step >= settings.episode_length:
                    state, episode_step = int(rng.integers(model.terminal)), 0
                else:
                    state = successor
        if settings.policy == "iteration":
            policy = _greedy(critic.q())
    return Experiment(settings, reward.copy(), snapshots, np.array(residuals),
                      np.array(changes), np.array(seconds), critic.weights.size,
                      0 if policy_phi is None else policy_phi.shape[1] * model.shape[1])


def evaluate_policy(model, reward, policy, discount):
    """Exact discounted return of a frozen policy on the gridded MDP."""
    n_states, n_actions = model.shape
    selector = csr_matrix((policy.ravel(), (np.repeat(np.arange(n_states), n_actions),
                                           np.arange(n_states * n_actions))),
                          shape=(n_states, n_states * n_actions))
    transition = (selector @ model.matrix)[:, :n_states]
    return spsolve(eye(n_states, format="csr") - discount * transition,
                   np.sum(policy * reward, axis=1))


def rollout(model, policy, initial_index, *, steps=100, seed=0):
    """Sample the selected frozen policy on the same MDP used for learning."""
    rng = np.random.default_rng(seed)
    state = int(initial_index)
    states, actions = [state], []
    for _ in range(steps):
        action = int(rng.choice(model.shape[1], p=policy[state]))
        actions.append(action)
        row = state * model.shape[1] + action
        lo, hi = model.matrix.indptr[row:row + 2]
        state = int(rng.choice(model.matrix.indices[lo:hi], p=model.matrix.data[lo:hi]))
        states.append(state)
        if state == model.terminal:
            break
    return np.array(states), np.array(actions)
