"""Fixed random-policy data and vectorized expressions for reward exploration."""

import numpy as np


def collect_rollouts(*, episodes=128, steps=600, dt=0.04,
                     torque_limit=1.0, walk_scale=0.15, seed=0,
                     kind="overdamped", damping=0.03,
                     initial_distribution="uniform", theta_bounds=(-np.pi, np.pi),
                     omega_bounds=(-1.0, 1.0), theta_mean=0.0, theta_std=0.5,
                     omega_mean=0.0, omega_std=0.5):
    """RK4 with held torque; random-walk increments are fractions of its limit.

    Overdamped: theta_dot = sin(theta) + u. Inertial (kind="inverted"):
    theta_ddot = sin(theta) - damping * theta_dot + u. No goal termination.
    Saved angles are unwrapped. Inertial states are [theta, omega].
    Initial coordinates are independent uniform or normal draws, or a fixed state.
    Normal angles are sampled unwrapped; displays use their circular equivalents.
    """
    if episodes < 1 or steps < 1 or int(episodes) != episodes or int(steps) != steps:
        raise ValueError("episodes and steps must be positive integers")
    if kind not in ("overdamped", "inverted"):
        raise ValueError("kind must be overdamped or inverted")
    if not np.all(np.isfinite([dt, torque_limit, walk_scale, damping])):
        raise ValueError("simulation parameters must be finite")
    if dt <= 0 or min(torque_limit, walk_scale, damping) < 0:
        raise ValueError("dt must be positive; torque limit, walk scale and damping nonnegative")
    rng = np.random.default_rng(seed)
    states = np.empty((episodes, steps + 1, 1 if kind == "overdamped" else 2))
    if initial_distribution not in ("uniform", "normal", "fixed"):
        raise ValueError("initial_distribution must be uniform, normal or fixed")

    def sample_coordinate(bounds, mean, std):
        if initial_distribution == "uniform":
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
                raise ValueError("uniform bounds must be a finite ordered pair")
            return rng.uniform(*bounds, episodes)
        if not np.isfinite(mean) or not np.isfinite(std) or std < 0:
            raise ValueError("normal/fixed means must be finite and standard deviations nonnegative")
        if initial_distribution == "fixed":
            return np.full(episodes, mean)
        return rng.normal(mean, std, episodes)

    states[:, 0, 0] = sample_coordinate(theta_bounds, theta_mean, theta_std)
    if kind == "inverted":
        states[:, 0, 1] = sample_coordinate(omega_bounds, omega_mean, omega_std)
    actions = np.empty((episodes, steps))
    torque = np.zeros(episodes)

    def rhs(state):
        if kind == "overdamped":
            return np.sin(state) + torque[:, None]
        theta, omega = state.T
        return np.column_stack((omega, np.sin(theta) - damping * omega + torque))

    for k in range(steps):
        torque = np.clip(torque + rng.normal(0, walk_scale * torque_limit, episodes),
                         -torque_limit, torque_limit)
        actions[:, k] = torque
        state = states[:, k]
        a = rhs(state)
        b = rhs(state + dt * a / 2)
        c = rhs(state + dt * b / 2)
        d = rhs(state + dt * c)
        states[:, k + 1] = state + dt * (a + 2 * b + 2 * c + d) / 6
    return {"states": states, "actions": actions, "dt": dt}


def sample_namespace(data, target=0.0):
    """One sample per post-transition state, paired with the torque just applied."""
    unwrapped = data["states"][:, 1:, 0].ravel()
    theta = (unwrapped + np.pi) % (2 * np.pi) - np.pi
    namespace = {
        "theta": theta,
        "theta_unwrapped": unwrapped,
        "u": data["actions"].ravel(),
        "t": np.tile(np.arange(1, data["actions"].shape[1] + 1) * data["dt"],
                     len(data["states"])),
        "target": target,
        "error": (theta - target + np.pi) % (2 * np.pi) - np.pi,
    }
    if data["states"].shape[-1] == 2:
        namespace["omega"] = data["states"][:, 1:, 1].ravel()
    return namespace


def evaluate_expression(source, namespace, size):
    """Evaluate a local NumPy expression, accepting scalar constants or one value/sample.

    Intended for the notebook author's Python expressions, not as a security sandbox.
    """
    try:
        with np.errstate(all="ignore"):
            result = np.asarray(eval(compile(source, "<expression>", "eval"),
                                     {"__builtins__": {}, "np": np}, namespace))
        if result.dtype.kind not in "biuf":
            raise ValueError("return real numbers or booleans")
        if result.shape not in ((), (size,)):
            raise ValueError(f"return a scalar or {size} values; got shape {result.shape}")
        result = np.broadcast_to(result.astype(float), (size,)).copy()
    except Exception as exc:
        raise ValueError(f"{type(exc).__name__}: {exc}") from exc
    bad = np.count_nonzero(~np.isfinite(result))
    if bad:
        raise ValueError(f"Expression produced {bad:,} nonfinite values out of {size:,}; "
                         "check divisions, square roots and logarithms.")
    return result
