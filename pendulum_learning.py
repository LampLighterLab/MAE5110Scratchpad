"""Small, inspectable numerical routines for the pendulum teaching notebooks.

Values are expected discounted rewards. Each transition receives a reward for
being in the captured equilibrium region and a negative quadratic penalty on
applied torque. There is no time cost or reward shaping. Arrival captures the
state; zero torque and equilibrium rewards continue through the finite horizon.
The actor-critic never receives a value-iteration or imitation initialization.
Angles are measured from upright: theta=0 is unstable, theta=+/-pi is downward,
and gravity contributes +sin(theta) to the overdamped velocity or acceleration.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Pendulum:
    kind: str = "overdamped"
    dt: float = 0.08
    torque_limit: float = 0.35
    horizon: int = 120
    target_theta: float = 0.06
    target_omega: float = 0.16
    omega_limit: float = 3.0
    torque_cost: float = 0.0
    discount: float = 0.98
    goal_reward: float = 1.0
    damping: float = 0.0

    def __post_init__(self):
        if self.kind not in ("overdamped", "undamped"):
            raise ValueError("kind must be overdamped or undamped")
        if self.dt <= 0 or self.horizon < 1 or self.torque_limit < 0:
            raise ValueError("dt and horizon must be positive; torque nonnegative")
        if not np.isfinite(self.torque_cost) or self.torque_cost < 0:
            raise ValueError("torque_cost must be finite and nonnegative")
        if not np.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if not np.isfinite(self.goal_reward) or self.goal_reward <= 0:
            raise ValueError("goal_reward must be finite and positive")
        if not np.isfinite(self.damping) or self.damping < 0:
            raise ValueError("damping must be finite and nonnegative")

    @property
    def dimension(self):
        return 1 if self.kind == "overdamped" else 2


def discount_sum(model, remaining):
    """Geometric sum 1 + gamma + ... + gamma**(remaining-1), including h=0."""
    remaining = np.asarray(remaining)
    if model.discount == 1:
        return remaining.astype(float)
    return -np.expm1(remaining * np.log(model.discount)) / (1 - model.discount)


def value_bounds(model, remaining):
    """Conservative discounted-return bounds from bounded per-step rewards."""
    weight = discount_sum(model, remaining)
    return (-model.torque_cost * model.torque_limit**2 * weight,
            model.goal_reward * weight)


def wrap_angle(theta):
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


def in_target(model, states):
    states = np.asarray(states)
    inside = np.abs(wrap_angle(states[..., 0])) <= model.target_theta
    if model.dimension == 2:
        inside &= np.abs(states[..., 1]) <= model.target_omega
    return inside


def step(model, states, actions):
    """RK4 with upright theta=0 and gravity +sin(theta); sampled-time capture."""
    states = np.asarray(states, dtype=float)
    actions = np.clip(np.asarray(actions), -model.torque_limit, model.torque_limit)

    def rhs(x):
        if model.dimension == 1:
            return (np.sin(x[..., 0]) + actions)[..., None]
        acceleration = np.sin(x[..., 0]) - model.damping * x[..., 1] + actions
        velocity = np.broadcast_to(x[..., 1], acceleration.shape)
        return np.stack((velocity, acceleration), axis=-1)

    a = rhs(states)
    b = rhs(states + 0.5 * model.dt * a)
    c = rhs(states + 0.5 * model.dt * b)
    d = rhs(states + model.dt * c)
    result = states + model.dt * (a + 2 * b + 2 * c + d) / 6
    result[..., 0] = wrap_angle(result[..., 0])
    return result


def policy_feature_options(dimension):
    """Display labels mapped to feature identifiers for a linear-in-weights policy."""
    options = {"θ": "theta"}
    if dimension == 2:
        options["θ̇"] = "omega"
    options.update({"sin θ": "sin_theta", "cos θ": "cos_theta",
                    "sin 2θ": "sin_2theta", "cos 2θ": "cos_2theta"})
    if dimension == 2:
        options.update({"θ̇ sin θ": "omega_sin_theta", "θ̇ cos θ": "omega_cos_theta",
                        "θ̇³": "omega_cubed", "E − E* = ½θ̇² + cos θ − 1": "energy_error",
                        "θ̇(E − E*)": "energy_pump"})
    return options


def policy_feature_names(model, feature_names=None):
    names = tuple(feature_names) if feature_names is not None else (
        ("theta",) if model.dimension == 1 else ("theta", "omega"))
    if not set(names).issubset(policy_feature_options(model.dimension).values()):
        raise ValueError("Feature unavailable for this state dimension")
    return names


def policy_features(model, states, feature_names=None):
    """State-only features; default is raw states. An empty basis gives zero torque.

    Upright energy error e=0.5*omega**2+cos(theta)-1 obeys
    de/dt=omega*u-damping*omega**2. The energy-pumping feature is omega*e,
    equivalently 0.5*omega**3 + omega*cos(theta) - omega.
    Angular velocity is not an independent state in the overdamped model.
    """
    states = np.asarray(states, dtype=float)
    names = policy_feature_names(model, feature_names)
    theta = wrap_angle(states[..., 0])
    columns = {"theta": theta, "sin_theta": np.sin(theta), "cos_theta": np.cos(theta),
               "sin_2theta": np.sin(2 * theta), "cos_2theta": np.cos(2 * theta)}
    if model.dimension == 2:
        omega = states[..., 1]
        error = 0.5 * omega**2 + np.cos(theta) - 1
        columns.update(omega=omega, omega_sin_theta=omega * np.sin(theta),
                       omega_cos_theta=omega * np.cos(theta), omega_cubed=omega**3,
                       energy_error=error, energy_pump=omega * error)
    if not names:
        return np.empty(states.shape[:-1] + (0,))
    return np.stack([columns[name] for name in names], axis=-1)


def policy_action(model, coefficients, states, feature_names=None):
    return np.clip(policy_features(model, states, feature_names) @ np.asarray(coefficients),
                   -model.torque_limit, model.torque_limit)


def initial_state_limits(model):
    """Half-widths of the local upright teaching distribution.

    With weak torque, the overdamped controllable basin is bounded by
    |theta| < asin(u_max). For the second-order model the local unstable
    coordinate is theta+omega; the default box keeps its magnitude below
    0.95*asin(u_max). The angular width exceeds the goal radius so rejection
    sampling can remove already captured starts. At actuator strengths too
    small for this neighborhood, the fallback still supplies non-goal starts
    but does not promise that they are controllable.
    """
    radius = np.arcsin(min(model.torque_limit, 0.95))
    if model.dimension == 1:
        return np.array([max(0.85 * radius, 1.1 * model.target_theta)])
    theta_width = max(0.65 * radius, 1.1 * model.target_theta)
    omega_width = 0.95 * radius - theta_width
    if omega_width <= 0:
        return np.array([1.25 * model.target_theta,
                         max(0.5 * model.target_omega, 0.25 * radius)])
    return np.array([theta_width, omega_width])


def initial_states(model, n=128, seed=0):
    """Uniform local-box starts conditioned on being outside the captured goal."""
    if n == 0:
        return np.empty((0, model.dimension))
    rng = np.random.default_rng(seed)
    limits = initial_state_limits(model)
    accepted = []
    count = 0
    while count < n:
        samples = rng.uniform(-limits, limits, size=(max(64, 2 * n), model.dimension))
        samples = samples[~in_target(model, samples)]
        accepted.append(samples)
        count += len(samples)
    return np.concatenate(accepted, axis=0)[:n]


def _rollout(model, initial, controller, seed, exploration):
    initial = np.asarray(initial, dtype=float).reshape(-1, model.dimension)
    n, h = len(initial), model.horizon
    rng = np.random.default_rng(seed)
    states = np.empty((n, h + 1, model.dimension))
    states[:, 0] = initial
    states[:, 0, 0] = wrap_angle(states[:, 0, 0])
    actions = np.zeros((n, h))
    latent = np.zeros((n, h))
    mean = np.zeros((n, h))
    alive = np.zeros((n, h), dtype=bool)
    hit_time = np.full(n, np.nan)
    hit_time[in_target(model, states[:, 0])] = 0
    active = np.isnan(hit_time)
    for t in range(h):
        alive[:, t] = active
        mean[:, t] = controller(states[:, t], h - t)
        latent[:, t] = mean[:, t] + exploration * rng.standard_normal(n)
        actions[:, t] = np.where(active, np.clip(latent[:, t],
                                  -model.torque_limit, model.torque_limit), 0)
        next_states = step(model, states[:, t], actions[:, t])
        states[:, t + 1] = np.where(active[:, None], next_states, states[:, t])
        newly_hit = active & in_target(model, states[:, t + 1])
        hit_time[newly_hit] = (t + 1) * model.dt
        active &= ~newly_hit
    goal_rewards = model.goal_reward * in_target(model, states[:, 1:])
    torque_rewards = -model.torque_cost * actions**2
    rewards = goal_rewards + torque_rewards
    returns = np.empty_like(rewards)
    tail = np.zeros(n)
    for t in range(h - 1, -1, -1):
        tail = rewards[:, t] + model.discount * tail
        returns[:, t] = tail
    weights = model.discount ** np.arange(h)
    return {"states": states, "actions": actions, "latent": latent,
            "mean": mean, "alive": alive, "returns": returns,
            "remaining": np.broadcast_to(np.arange(h, 0, -1), (n, h)),
            "return": returns[:, 0], "success": ~active,
            "elapsed_time": model.dt * alive.sum(axis=1),
            "goal_return": goal_rewards @ weights,
            "torque_return": torque_rewards @ weights,
            "rewards": rewards, "goal_rewards": goal_rewards,
            "torque_rewards": torque_rewards,
            "hit_time": hit_time, "exploration": exploration}


def rollout(model, coefficients, initial_states, seed=0, exploration=0.0, feature_names=None):
    """Gaussian *latent* action z~N(phi k,sigma²), executed torque clip(z).

    States have shape [episode,time,dimension], including H+1 sample times.
    All step arrays have shape [episode,H]. `alive` marks pre-capture decisions;
    rewards keep accruing after capture. `returns` is the exact relative-time
    discounted MC reward-to-go. `return` is its time-zero value per episode,
    decomposed into goal_return and negative torque_return. Elapsed time is
    diagnostic only and never appears in the reward.
    """
    coefficients = np.asarray(coefficients)
    return _rollout(model, initial_states,
                    lambda states, remaining: policy_features(model, states, feature_names) @ coefficients,
                    seed, exploration)


def _interpolation(model, theta, omega, states):
    """Indices and weights for periodic theta / bounded omega interpolation."""
    ti = np.mod(states[..., 0] - theta[0], 2 * np.pi) * len(theta) / (2 * np.pi)
    t0 = np.floor(ti).astype(int) % len(theta)
    t1 = (t0 + 1) % len(theta)
    wt = ti - np.floor(ti)
    if model.dimension == 1:
        return (t0, t1, wt)
    wi = (states[..., 1] - omega[0]) / (omega[1] - omega[0])
    wi = np.clip(wi, 0, len(omega) - 1)
    w0 = np.minimum(np.floor(wi).astype(int), len(omega) - 2)
    w1 = w0 + 1
    return (t0, t1, wt, w0, w1, wi - w0)


def _interpolate(values, indices):
    t0, t1, wt = indices[:3]
    if len(indices) == 3:
        return (1 - wt) * values[t0] + wt * values[t1]
    w0, w1, ww = indices[3:]
    return ((1 - wt) * ((1 - ww) * values[t0, w0] + ww * values[t0, w1])
            + wt * ((1 - ww) * values[t1, w0] + ww * values[t1, w1]))


def value_iteration(model, n_theta=161, n_omega=81, n_actions=15,
                    snapshot_steps=None):
    """Bellman recursion: V_0=0, V_h=max_a [r(s,a,s')+gamma*V_{h-1}].

    Target capture is absorbing with value R*sum(gamma**j,j=0..h-1).
    Values[h] and policy[h] are
    indexed by *remaining* steps, not optimization iterations. Theta is
    periodic. An undamped transition outside the omega grid receives the
    conservative lower return bound; no clipped boundary becomes a shortcut.
    Policy arrays contain actual torque, not action indices.
    """
    # Include theta=0 exactly for odd as well as even periodic grid sizes.
    theta = (np.arange(n_theta) - n_theta // 2) * (2 * np.pi / n_theta)
    omega = np.linspace(-model.omega_limit, model.omega_limit, n_omega)
    if model.dimension == 1:
        grid = theta[:, None]
        shape = (n_theta,)
    else:
        tg, wg = np.meshgrid(theta, omega, indexing="ij")
        grid = np.stack((tg, wg), axis=-1)
        shape = (n_theta, n_omega)
    flat = grid.reshape(-1, model.dimension)
    actions = np.linspace(-model.torque_limit, model.torque_limit, n_actions)
    # Stable tie breaking prefers low effort when predicted returns are equal.
    actions = actions[np.argsort(np.abs(actions), kind="stable")]
    transitions = step(model, flat[:, None, :], actions[None, :])
    indices = _interpolation(model, theta, omega, transitions)
    terminal = in_target(model, transitions)
    outside = (np.abs(transitions[..., 1]) > model.omega_limit
               if model.dimension == 2 else np.zeros(terminal.shape, dtype=bool))
    target = in_target(model, grid)
    values = np.zeros((model.horizon + 1,) + shape, dtype=np.float32)
    policy = np.zeros_like(values)
    for h in range(1, model.horizon + 1):
        continuation = _interpolate(values[h - 1], indices)
        lower, upper = value_bounds(model, h - 1)
        continuation = np.where(terminal, upper, continuation)
        continuation = np.where(outside, lower, continuation)
        rewards = model.goal_reward * terminal - model.torque_cost * actions[None, :]**2
        q = rewards + model.discount * continuation
        choice = q.argmax(axis=1)
        values[h] = q[np.arange(len(flat)), choice].reshape(shape)
        policy[h] = actions[choice].reshape(shape)
        values[h][target] = model.goal_reward * discount_sum(model, h)
        policy[h][target] = 0
    if snapshot_steps is None:
        snapshot_steps = sorted(set([1, model.horizon // 4, model.horizon // 2,
                                     model.horizon]))
    return {"theta": theta, "omega": omega if model.dimension == 2 else None,
            "values": values, "policy": policy, "actions": actions,
            "target_mask": target, "overflow_fraction": float(outside.mean()),
            "snapshots": {h: {"value": values[h], "policy": policy[h]}
                          for h in snapshot_steps if 0 <= h <= model.horizon}}


def vi_action(model, vi, states, remaining):
    """One-step reward maximization using interpolated grid values."""
    if remaining <= 0:
        return np.zeros(np.asarray(states).shape[:-1])
    states = np.asarray(states)
    transitions = step(model, states[..., None, :], vi["actions"])
    indices = _interpolation(model, vi["theta"], vi["omega"], transitions)
    q = _interpolate(vi["values"][remaining - 1], indices)
    terminal = in_target(model, transitions)
    lower, upper = value_bounds(model, remaining - 1)
    q = np.where(terminal, upper, q)
    if model.dimension == 2:
        q = np.where(np.abs(transitions[..., 1]) > model.omega_limit,
                     lower, q)
    q = (model.goal_reward * terminal - model.torque_cost * vi["actions"]**2
         + model.discount * q)
    return np.where(in_target(model, states), 0, vi["actions"][q.argmax(axis=-1)])


def vi_rollout(model, vi, initial_states):
    return _rollout(model, initial_states,
                    lambda states, remaining: vi_action(model, vi, states, remaining),
                    seed=0, exploration=0)


def vi_policy_rollout(model, vi, initial_states, iteration):
    """Run one saved Bellman policy for the full observation horizon.

    Unlike vi_rollout, the policy index stays fixed throughout each orbit.
    Iteration zero is the explicitly zero-initialized policy.
    """
    if int(iteration) != iteration or not 0 <= iteration < len(vi["values"]):
        raise ValueError("Iteration must index an available Bellman snapshot")
    return _rollout(model, initial_states,
                    lambda states, remaining: vi_action(model, vi, states, int(iteration)),
                    seed=0, exploration=0)


def collect_policy_data(model, vi, n_samples=3000, seed=3, remaining=None,
                        source="rollouts"):
    """State/action labels from one Bellman snapshot, reusable across features.

    `grid` includes every displayed grid state once and uses its saved
    action exactly. `rollouts` samples visited states from local demonstrations;
    their occupancy distribution is generally very different from the grid.
    """
    iteration = model.horizon if remaining is None else remaining
    if int(iteration) != iteration or not 0 <= iteration < len(vi["policy"]):
        raise ValueError("Iteration must index an available Bellman snapshot")
    iteration = int(iteration)
    if source == "grid":
        if model.dimension == 1:
            states = vi["theta"][:, None]
        else:
            theta, omega = np.meshgrid(vi["theta"], vi["omega"], indexing="ij")
            states = np.stack([theta, omega], axis=-1).reshape(-1, 2)
        return {"states": states, "actions": vi["policy"][iteration].ravel(),
                "values": vi["values"][iteration].ravel(),
                "batch": None, "iteration": iteration, "source": source}
    if source != "rollouts":
        raise ValueError("Data source must be grid or rollouts")
    batch = vi_policy_rollout(model, vi, initial_states(model, 192, seed), iteration)
    states = batch["states"][:, :-1][batch["alive"]]
    actions = batch["actions"][batch["alive"]]
    if len(states) > n_samples:
        chosen = np.random.default_rng(seed + 1).choice(len(states), n_samples, replace=False)
        states, actions = states[chosen], actions[chosen]
    values = _interpolate(vi["values"][iteration],
                          _interpolation(model, vi["theta"], vi["omega"], states))
    if model.dimension == 2:
        values = np.where(np.abs(states[:, 1]) > model.omega_limit,
                          value_bounds(model, iteration)[0], values)
    return {"states": states, "actions": actions, "batch": batch,
            "values": values, "iteration": iteration, "source": source}


def value_policy_weights(values, power=1.0):
    """Emphasize positive-value states: w = (max(V, 0) / max(max(V, 0)))**p.

    p=0 is uniform; p=1 is proportional to positive value. Higher powers
    concentrate on the best states. All-nonpositive values use uniform weights,
    including the zero-initialized Bellman iterate. No future values are used.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Values must be a nonempty, finite vector")
    if not np.isfinite(power) or power < 0:
        raise ValueError("Value-weight exponent must be finite and nonnegative")
    positive = np.maximum(values, 0)
    if power == 0 or positive.max() == 0:
        return np.ones_like(positive)
    return (positive / positive.max())**power


def fit_linear_policy(model, vi, remaining=None, n_samples=3000, seed=3,
                      feature_names=None, data=None, sample_weights=None):
    """Least squares in chosen features, followed by clipping at execution.

    `remaining` selects a saved Bellman snapshot, including zero initialization.
    Every label comes from that fixed policy, with no later iterates used. Feature columns
    are scaled before the SVD solve to handle different units and dependencies.
    Optional nonnegative sample weights minimize sum(w * (phi k - u)**2);
    square roots of the weights multiply both the design rows and labels.
    """
    data = (collect_policy_data(model, vi, n_samples, seed, remaining)
            if data is None else data)
    iteration = data.get("iteration", model.horizon if remaining is None else remaining)
    if remaining is not None and remaining != iteration:
        raise ValueError("Demonstration data must match the requested Bellman snapshot")
    states, actions, batch = data["states"], data["actions"], data["batch"]
    names = policy_feature_names(model, feature_names)
    features = policy_features(model, states, names)
    sample_weights = (np.ones(len(states)) if sample_weights is None
                      else np.asarray(sample_weights, dtype=float))
    if (sample_weights.shape != (len(states),) or not np.isfinite(sample_weights).all()
            or np.any(sample_weights < 0) or not np.any(sample_weights > 0)):
        raise ValueError("Sample weights must be finite, nonnegative, nonzero, and match the data")
    sample_weights = sample_weights / sample_weights.max()
    root_weights = np.sqrt(sample_weights)
    design = features * root_weights[:, None]
    scale = np.maximum(np.linalg.norm(design, axis=0), 1e-12)
    scaled_coefficients, _, rank, singular_values = np.linalg.lstsq(
        design / scale, actions * root_weights, rcond=None,
    )
    coefficients = scaled_coefficients / scale
    raw_predictions = features @ coefficients
    predictions = policy_action(model, coefficients, states, names)
    condition = (float(singular_values[0] / singular_values[rank - 1])
                 if rank else 0.0)
    return {"coefficients": coefficients, "states": states, "actions": actions,
            "predictions": predictions, "rmse": float(np.sqrt(np.mean((predictions-actions)**2))),
            "raw_rmse": float(np.sqrt(np.mean((raw_predictions-actions)**2))),
            "weighted_rmse": float(np.sqrt(np.average(
                (raw_predictions-actions)**2, weights=sample_weights))),
            "sample_weights": sample_weights,
            "condition": condition, "source": data.get("source", "custom"),
            "batch": batch, "feature_names": names, "rank": int(rank),
            "iteration": iteration}


def value_features(model, states, remaining):
    """Even polynomial state features crossed with [1,r,r²], r=remaining/H.

    In two dimensions theta*omega captures coupled position/velocity dependence.
    Global sign symmetry V(-x)=V(x) is built in, but no optimal solution is used.
    """
    states = policy_features(model, states)
    theta = states[..., 0] / np.pi
    one = np.ones_like(theta)
    if model.dimension == 1:
        spatial = np.stack((one, np.abs(theta), theta**2,
                            np.abs(theta)**3, theta**4), axis=-1)
    else:
        omega = states[..., 1] / model.omega_limit
        spatial = np.stack((one, np.abs(theta), np.abs(omega), theta**2,
                            theta*omega, omega**2, np.abs(theta)**3,
                            np.abs(omega)**3, theta**4, omega**4,
                            theta**2*omega**2), axis=-1)
    r = np.broadcast_to(np.asarray(remaining) / model.horizon, theta.shape)
    time = np.stack((one, r, r**2), axis=-1)
    return (spatial[..., :, None] * time[..., None, :]).reshape(
        theta.shape + (spatial.shape[-1] * time.shape[-1],))


def fit_value(model, batch, ridge=1e-5):
    """Ridge least squares against exact discounted on-policy MC returns."""
    mask = batch["alive"]
    states = batch["states"][:, :-1][mask]
    remaining = batch["remaining"][mask]
    targets = batch["returns"][mask]
    features = value_features(model, states, remaining)
    if len(targets) == 0:
        return {"coefficients": np.zeros(features.shape[-1]), "ridge": ridge,
                "n_samples": 0, "rmse": 0.0}
    # Normalize columns so one ridge parameter treats feature scales fairly.
    scale = np.sqrt(np.mean(features**2, axis=0)) + 1e-8
    x = features / scale
    weights = np.linalg.solve(x.T @ x + ridge * len(x) * np.eye(x.shape[1]), x.T @ targets)
    coefficients = weights / scale
    critic = {"coefficients": coefficients, "ridge": ridge, "n_samples": len(targets)}
    predictions = predict_value(model, critic, states, remaining)
    critic["rmse"] = float(np.sqrt(np.mean((predictions - targets)**2)))
    return critic


def predict_value(model, critic, states, remaining):
    values = value_features(model, states, remaining) @ critic["coefficients"]
    lower, upper = value_bounds(model, remaining)
    values = np.clip(values, lower, upper)
    return np.where(in_target(model, states), upper, values)


def _metrics(batch):
    hits = batch["hit_time"][batch["success"]]
    return {"mean_return": float(batch["return"].mean()),
            "mean_goal_return": float(batch["goal_return"].mean()),
            "mean_torque_return": float(batch["torque_return"].mean()),
            "mean_elapsed_time": float(batch["elapsed_time"].mean()),
            "success_rate": float(batch["success"].mean()),
            "mean_hit_time": float(hits.mean()) if len(hits) else float("nan")}


def evaluate_policy(model, coefficients, n_episodes=256, seed=41, exploration=0):
    return _metrics(rollout(model, coefficients, initial_states(model, n_episodes, seed),
                            seed=seed+1, exploration=exploration))


def train_actor_critic(model, iterations=25, batch_size=128, seed=7,
                       exploration=0.18, learning_rate=0.22):
    """On-policy MC actor-critic with a natural score-gradient reward ascent.

    A previous-batch critic is an action-independent baseline. For the first
    update, leave-one-episode-out baselines at each time give that independence.
    The new critic is fitted and saved with the policy that generated its data.
    The actor uses z (before clipping): score=phi*(z-phi k)/sigma².
    The gradient of the time-zero objective includes gamma**t multiplying each
    score times its relative-time advantage. Discount-weighted empirical
    Fisher preconditioning is followed by a bounded ascent step. No VI
    result, demonstration, or analytic damping gain enters this routine.
    """
    if exploration <= 0:
        raise ValueError("Actor learning needs positive Gaussian exploration")
    rng = np.random.default_rng(seed)
    coefficients = rng.uniform(-0.15, 0.15, size=model.dimension)
    snapshots, history = [], []
    previous_critic = None
    evaluation_starts = initial_states(model, 256, seed=41)
    first_batch = None
    for iteration in range(iterations + 1):
        starts = initial_states(model, batch_size, seed=seed + 1000 + iteration)
        batch = rollout(model, coefficients, starts, seed=seed + 2000 + iteration,
                        exploration=exploration)
        if first_batch is None:
            first_batch = batch
        critic = fit_value(model, batch)
        metrics = _metrics(rollout(model, coefficients, evaluation_starts))
        metrics.update({"mean_return_stochastic": float(batch["return"].mean()),
                        "mean_goal_return_stochastic": float(batch["goal_return"].mean()),
                        "mean_torque_return_stochastic": float(batch["torque_return"].mean()),
                        "success_rate_stochastic": float(batch["success"].mean()),
                        "critic_rmse": critic["rmse"], "iteration": iteration})
        history.append(metrics)
        snapshots.append({"iteration": iteration, "coefficients": coefficients.copy(),
                          "critic": critic, "metrics": metrics})
        if iteration == iterations:
            break
        mask = batch["alive"]
        states = batch["states"][:, :-1][mask]
        features = policy_features(model, states)
        returns = batch["returns"][mask]
        if previous_critic is None:
            active_returns = batch["returns"] * mask
            sums = active_returns.sum(axis=0, keepdims=True)
            counts = batch["alive"].sum(axis=0, keepdims=True)
            baseline_all = (sums - active_returns) / np.maximum(counts - 1, 1)
            baseline = baseline_all[mask]
        else:
            baseline = predict_value(model, previous_critic, states, batch["remaining"][mask])
        advantage = returns - baseline
        score = features * ((batch["latent"][mask] - batch["mean"][mask]) /
                            exploration**2)[:, None]
        weights = model.discount ** (model.horizon - batch["remaining"][mask])
        gradient = np.mean(weights[:, None] * score * advantage[:, None], axis=0)
        fisher = features.T @ (weights[:, None] * features) / (len(features) * exploration**2)
        direction = np.linalg.solve(fisher + 1e-3 * np.eye(model.dimension), gradient)
        direction /= max(np.linalg.norm(direction), 1e-8)
        coefficients = coefficients + learning_rate * direction
        previous_critic = critic
    return {"coefficients": coefficients, "history": history, "snapshots": snapshots,
            "initial_batch": first_batch, "final_batch": batch,
            "exploration": exploration, "learning_rate": learning_rate, "seed": seed}
