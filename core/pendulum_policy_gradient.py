"""Small numerical core for a two-parameter pendulum policy-gradient lesson.

The physical angle is unwrapped, with the particular upright origin as the
goal.  The controller is continuous feedback, with no torque clipping:

    u = w1 * omega + w2 * sin(theta) - theta - omega.

Only w1 and w2 are learned.  At (w1, w2) = (b, -1), the closed-loop equation
is exactly theta'' + theta' + theta = 0.  Rewards are evaluated after every
integration step; arriving in the goal never freezes the dynamics.
"""

from dataclasses import asdict, dataclass
import inspect

import numpy as np


DEFAULT_REWARD_SOURCE = """def reward(state, action, next_state, params):
    near_theta = np.isclose(next_state[..., 0], 0.0, atol=params["theta_tolerance"], rtol=0.0)
    near_omega = np.isclose(next_state[..., 1], 0.0, atol=params["omega_tolerance"], rtol=0.0)
    return 1.0 * (near_theta & near_omega)
"""


class RewardError(ValueError):
    """An editable reward definition could not be compiled or evaluated."""


@dataclass(frozen=True)
class PendulumConfig:
    b: float = 0.03
    dt: float = 0.04
    horizon: int = 250
    discount: float = 0.98
    theta_tolerance: float = 0.05
    omega_tolerance: float = 0.05
    theta_width: float = 1.0
    omega_width: float = 1.0

    def __post_init__(self):
        for name in ("b", "dt", "discount", "theta_tolerance",
                     "omega_tolerance", "theta_width", "omega_width"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.b < 0:
            raise ValueError("b must be nonnegative")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if not isinstance(self.horizon, (int, np.integer)) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if self.theta_tolerance <= 0 or self.omega_tolerance <= 0:
            raise ValueError("goal tolerances must be positive")
        if self.theta_width <= 0 or self.omega_width <= 0:
            raise ValueError("initial-state half-widths must be positive")
        if (self.theta_width <= self.theta_tolerance
                and self.omega_width <= self.omega_tolerance):
            raise ValueError("initial-state box must contain states outside the goal")

    @property
    def duration(self):
        return self.dt * self.horizon

    @property
    def cancellation_weights(self):
        return np.array([self.b, -1.0])


def isnear(theta, omega, config=None):
    """Compatibility helper using np.isclose at the unwrapped origin."""
    if config is None:
        config = PendulumConfig()
    return (np.isclose(theta, 0.0, atol=config.theta_tolerance, rtol=0.0)
            & np.isclose(omega, 0.0, atol=config.omega_tolerance, rtol=0.0))


def goal_mask(states, config):
    states = np.asarray(states, dtype=float)
    return isnear(states[..., 0], states[..., 1], config)


def _reward_values(function, state, action, next_state, params, t):
    """Evaluate one batch of transitions; allow a scalar constant reward."""
    try:
        values = np.asarray(function(state, action, next_state, params.copy()), dtype=float)
        target_shape = state.shape[:-1]
        if values.shape != target_shape and values.ndim != 0:
            raise RewardError(
                f"reward must return a scalar or shape {target_shape}; got {values.shape}"
            )
        values = np.broadcast_to(values, target_shape)
    except RewardError:
        raise
    except Exception as error:
        raise RewardError(f"reward failed at t={t:g}: {type(error).__name__}: {error}") from error
    if not np.all(np.isfinite(values)):
        raise RewardError(f"reward returned a nonfinite value at t={t:g}")
    return values


def compile_reward(source, config):
    """Compile the editor's vectorized ``reward(state, action, next_state, params)``.

    ``params`` is a dictionary containing all PendulumConfig fields. ``np``
    is available as a global. State arrays have shape ``(..., 2)``, ordered theta, omega;
    actions and returned rewards have shape ``(...)``.  Scalar constant
    rewards broadcast over the batch.  Evaluation is entirely vectorized.
    This executes local Python code; it is not a security sandbox.
    """
    if not isinstance(source, str) or not source.strip():
        raise RewardError("Define a Python function: def reward(state, action, next_state, params): ...")
    try:
        code = compile(source, "<editable reward>", "exec")
    except SyntaxError as error:
        raise RewardError(f"Reward syntax error on line {error.lineno}: {error.msg}") from error
    namespace = {
        "np": np,
        "theta_atol": config.theta_tolerance,
        "omega_atol": config.omega_tolerance,
        "isnear": lambda theta, omega: isnear(theta, omega, config),
    }
    try:
        exec(code, namespace)
    except Exception as error:
        raise RewardError(f"Reward definition failed: {type(error).__name__}: {error}") from error
    function = namespace.get("reward")
    if not callable(function):
        raise RewardError("The code must define a function named reward(state, action, next_state, params)")
    try:
        inspect.signature(function).bind(np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2)), asdict(config))
    except (TypeError, ValueError) as error:
        raise RewardError("reward must accept four arguments: state, action, next_state, params") from error
    _reward_values(function, np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2)), asdict(config), 0.0)
    return function


def _as_reward(reward):
    if not callable(reward):
        raise RewardError("reward must be callable")
    return reward


def sample_initial_states(config, n=128, seed=0):
    """Uniform states in the configured box, conditioned on starting outside goal."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    limits = np.array([config.theta_width, config.omega_width])
    accepted = []
    remaining = n
    while remaining:
        candidates = rng.uniform(-limits, limits, size=(max(64, 2 * remaining), 2))
        candidates = candidates[~goal_mask(candidates, config)]
        chosen = candidates[:remaining]
        accepted.append(chosen)
        remaining -= len(chosen)
    return np.concatenate(accepted, axis=0)


def _validate_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.shape[-1:] != (2,) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must have a final dimension of two finite values")
    return weights


def _validate_starts(starts):
    starts = np.asarray(starts, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 2 or len(starts) < 1:
        raise ValueError("starts must be a nonempty array with shape (episodes, 2)")
    if not np.all(np.isfinite(starts)):
        raise ValueError("initial states must be finite")
    return starts


def policy_torque(weights, states):
    """Unbounded torque at the unwrapped physical state."""
    weights = np.asarray(weights, dtype=float)
    states = np.asarray(states, dtype=float)
    theta, omega = states[..., 0], states[..., 1]
    return weights[..., 0] * omega + weights[..., 1] * np.sin(theta) - theta - omega


def closed_loop_rhs(states, weights, config):
    """Physical derivative under the fixed-PD, two-parameter controller."""
    states = np.asarray(states, dtype=float)
    theta, omega = states[..., 0], states[..., 1]
    acceleration = np.sin(theta) - config.b * omega + policy_torque(weights, states)
    return np.stack((omega, acceleration), axis=-1)


def step(states, weights, config, *, passive=False):
    """Batched RK4 for continuous feedback, including feedback at every substage."""
    states = np.asarray(states, dtype=float)

    def rhs(state):
        if passive:
            theta, omega = state[..., 0], state[..., 1]
            return np.stack((omega, np.sin(theta) - config.b * omega), axis=-1)
        return closed_loop_rhs(state, weights, config)

    with np.errstate(over="ignore", invalid="ignore"):
        a = rhs(states)
        b = rhs(states + 0.5 * config.dt * a)
        c = rhs(states + 0.5 * config.dt * b)
        d = rhs(states + config.dt * c)
        result = states + config.dt * (a + 2 * b + 2 * c + d) / 6
    if not np.all(np.isfinite(result)):
        raise ValueError("Pendulum integration diverged; use smaller policy weights or a smaller dt")
    return result


def evaluate_rollouts(weights, starts, config, reward, *, passive=False):
    """Return saved trajectories and discounted Monte Carlo returns.

    ``states`` has shape (episodes, horizon + 1, 2).  ``actions``, ``rewards``
    and relative-time reward-to-go ``returns`` have shape (episodes, horizon).
    ``return`` is each trajectory's time-zero discounted sum.  The reward gets
    the pre-transition state, torque sampled there, and post-transition state
    as one vectorized call per integration step.  Success means ever entering
    the goal.
    """
    weights = _validate_weights(weights)
    if weights.shape != (2,):
        raise ValueError("evaluate_rollouts expects one weight pair with shape (2,)")
    starts = _validate_starts(starts)
    reward = _as_reward(reward)
    params = asdict(config)
    n, h = len(starts), config.horizon
    states = np.empty((n, h + 1, 2), dtype=float)
    states[:, 0] = starts
    actions = np.empty((n, h), dtype=float)
    rewards = np.empty((n, h), dtype=float)
    hit_time = np.where(goal_mask(starts, config), 0.0, np.nan)
    for index in range(h):
        actions[:, index] = 0.0 if passive else policy_torque(weights, states[:, index])
        states[:, index + 1] = step(states[:, index], weights, config, passive=passive)
        time = (index + 1) * config.dt
        rewards[:, index] = _reward_values(
            reward, states[:, index], actions[:, index], states[:, index + 1], params, time,
        )
        newly_hit = np.isnan(hit_time) & goal_mask(states[:, index + 1], config)
        hit_time[newly_hit] = time
    returns = np.empty_like(rewards)
    tail = np.zeros(n)
    for index in range(h - 1, -1, -1):
        tail = rewards[:, index] + config.discount * tail
        returns[:, index] = tail
    if not np.all(np.isfinite(returns)):
        raise RewardError("reward magnitudes made the discounted return nonfinite")
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
        "return": returns[:, 0],
        "score": float(np.mean(returns[:, 0])),
        "success": ~np.isnan(hit_time),
        "hit_time": hit_time,
        "final_success": goal_mask(states[:, -1], config),
        "time": np.arange(h + 1) * config.dt,
    }


def passive_rollouts(starts, config, reward=None):
    """The slightly damped pendulum with u=0, for the introductory phase plot."""
    if reward is None:
        reward = compile_reward(DEFAULT_REWARD_SOURCE, config)
    return evaluate_rollouts([0.0, 0.0], starts, config, reward, passive=True)


def evaluate_parameters(parameter_pairs, starts, config, reward, chunk_size=256,
                        *, return_episode_returns=False):
    """Mean discounted return for many policies, without storing trajectories.

    A common starting batch is used for every policy.  Leading parameter-array
    dimensions are preserved, so (rows, columns, 2) returns a landscape matrix.
    Chunking bounds memory while integrating parameters and episodes together.
    ``return_episode_returns=True`` retains the individual Monte Carlo returns
    as the final axis, for uncertainty estimates in finite differences.
    """
    parameters = _validate_weights(parameter_pairs)
    original_shape = parameters.shape[:-1]
    parameters = parameters.reshape(-1, 2)
    starts = _validate_starts(starts)
    reward = _as_reward(reward)
    params = asdict(config)
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    scores = np.empty((len(parameters), len(starts)), dtype=float)
    discounts = config.discount ** np.arange(config.horizon)
    for first in range(0, len(parameters), chunk_size):
        chunk = parameters[first:first + chunk_size]
        weights = chunk[:, None, :]
        state = np.broadcast_to(starts, (len(chunk),) + starts.shape).copy()
        totals = np.zeros(state.shape[:-1], dtype=float)
        for index, discount in enumerate(discounts):
            torque = policy_torque(weights, state)
            next_state = step(state, weights, config)
            totals += discount * _reward_values(
                reward, state, torque, next_state, params,
                (index + 1) * config.dt,
            )
            state = next_state
        if not np.all(np.isfinite(totals)):
            raise RewardError("reward magnitudes made the discounted return nonfinite")
        scores[first:first + len(chunk)] = totals
    if return_episode_returns:
        return scores.reshape(original_shape + (len(starts),))
    return np.mean(scores, axis=1).reshape(original_shape)


def train_policy_gradient(initial_weights, config, reward, iterations=8,
                          batch_size=32, seed=0, epsilon=0.15,
                          learning_rate=0.05, resample=False):
    """Central-coordinate finite differences followed by plain gradient ascent.

    There is no critic, exploration noise, torque bound, gradient clipping, or
    policy projection.  All four perturbations share exactly the same starting
    batch.  By default that seeded batch is also held fixed across updates;
    ``resample=True`` instead draws a new batch per snapshot.  A separate fixed
    batch supplies comparable evaluation scores.  Snapshot zero is the
    initial policy; snapshot k is
    the policy after k updates.  Every snapshot includes its gradient estimate
    (including the last, which is displayed but not applied).
    """
    weights = _validate_weights(initial_weights).copy()
    if weights.shape != (2,):
        raise ValueError("initial_weights must have shape (2,)")
    if not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not isinstance(resample, (bool, np.bool_)):
        raise ValueError("resample must be a boolean")
    rng = np.random.default_rng(seed)
    evaluation_seed = np.random.SeedSequence(seed).spawn(1)[0]
    evaluation_starts = sample_initial_states(config, batch_size, evaluation_seed)
    fixed_starts = sample_initial_states(config, batch_size, rng)
    snapshots = []
    rollout_count = 0
    for iteration in range(iterations + 1):
        starts = (sample_initial_states(config, batch_size, rng)
                  if resample and iteration > 0 else fixed_starts)
        offsets = epsilon * np.eye(2)
        perturbed = np.concatenate((weights + offsets, weights - offsets), axis=0)
        perturbation_rollouts = [
            evaluate_rollouts(probe, starts, config, reward) for probe in perturbed
        ]
        episode_returns = np.stack([batch["return"] for batch in perturbation_rollouts])
        scores = np.mean(episode_returns, axis=1)
        plus_scores, minus_scores = scores[:2], scores[2:]
        gradient = (plus_scores - minus_scores) / (2 * epsilon)
        sample_gradients = (episode_returns[:2] - episode_returns[2:]) / (2 * epsilon)
        gradient_stderr = (np.std(sample_gradients, axis=1, ddof=1) / np.sqrt(batch_size)
                           if batch_size > 1 else np.zeros(2))
        rollout = evaluate_rollouts(weights, starts, config, reward)
        evaluation_rollout = evaluate_rollouts(weights, evaluation_starts, config, reward)
        evaluation_score = evaluation_rollout["score"]
        next_weights = weights + learning_rate * gradient if iteration < iterations else weights.copy()
        if not np.all(np.isfinite(next_weights)):
            raise ValueError("The policy-gradient update became nonfinite; reduce the learning rate")
        rollout_count += 6 * batch_size
        snapshots.append({
            "iteration": iteration,
            "weights": weights.copy(),
            "gradient": gradient.copy(),
            "gradient_stderr": gradient_stderr.copy(),
            "plus_scores": plus_scores.copy(),
            "minus_scores": minus_scores.copy(),
            "perturbed_weights": perturbed.copy(),
            "perturbation_rollouts": perturbation_rollouts,
            "batch_starts": starts.copy(),
            "rollout": rollout,
            "batch_score": rollout["score"],
            "evaluation_score": evaluation_score,
            "evaluation_rollout": evaluation_rollout,
            "next_weights": next_weights.copy(),
            "epsilon": float(epsilon),
            "learning_rate": float(learning_rate),
            "rollout_count": rollout_count,
            "step_count": rollout_count * config.horizon,
        })
        weights = next_weights
    return {
        "snapshots": snapshots,
        "evaluation_starts": evaluation_starts,
        "training_starts": fixed_starts,
        "seed": seed,
        "resample": bool(resample),
        "rollout_count": rollout_count,
        "step_count": rollout_count * config.horizon,
        "initial_weights": snapshots[0]["weights"].copy(),
        "final_weights": weights.copy(),
    }
