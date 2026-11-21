"""Discrete double-integrator tracking and equivalent finite-horizon QPs."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import BPoly
from scipy.linalg import solve_discrete_are
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize

# Position and velocity are measured in m and m/s, acceleration in m/s².
Q = np.diag([4., 1.])
R = 0.06
DURATION = 4.


def rollout(A, B, initial, actions):
    states = [initial]
    for action in actions:
        states.append(A @ states[-1] + B * action)
    return np.array(states)


def tracking_gain(A, B):
    """Steady-state discrete LQR gain for state error and feedforward input deviation."""
    P = solve_discrete_are(A, B[:, None], Q, [[R]])
    return (B @ P @ A) / (R + B @ P @ B)


@dataclass
class Model:
    time: np.ndarray
    A: np.ndarray
    B: np.ndarray
    reference: np.ndarray
    forcing: np.ndarray
    feedforward: np.ndarray
    curve_time: np.ndarray
    curve: np.ndarray


def reference_motion(amplitude, dt):
    """Move from rest to amplitude and back to amplitude/2, at rest at each waypoint."""
    per_second = round(1 / dt)
    if per_second < 1 or not np.isclose(dt * per_second, 1):
        raise ValueError("Choose a time step that divides one second.")
    time = np.arange(round(DURATION * per_second) + 1) * dt
    A = np.array([[1., dt], [0., 1.]])
    B = np.array([dt**2 / 2, dt])
    # Sample a raised cosine in position. The sine velocity scale makes
    # p[k+1] - p[k] = dt*(v[k]+v[k+1])/2 exactly on each selected grid.
    omega = 2*np.pi / DURATION
    turn = len(time)//2
    returning = time > time[turn]
    leg_distance = np.where(returning, -amplitude/2, amplitude)
    leg_phase = omega * np.where(returning, time-time[turn], time)
    position = np.where(returning, amplitude, 0.) + 0.5*leg_distance*(1 - np.cos(leg_phase))
    velocity = leg_distance*np.tan(omega*dt/2)/dt * np.sin(leg_phase)
    velocity[[0, turn, -1]] = 0.
    reference = np.column_stack([position, velocity])
    # A quintic Hermite segment matches p, v, and acceleration at both ends.
    # The curve is C² and its plotted velocity is the derivative of position.
    acceleration = 0.5*leg_distance*omega**2*np.cos(leg_phase)
    # Use the mean of the two leg curvatures for a common turnaround acceleration.
    acceleration[turn] = -0.5*omega**2 * (amplitude + amplitude/2)/2
    acceleration[[0, -1]] = 0.  # Join to stationary motion before and after.
    smooth = BPoly.from_derivatives(time, np.column_stack([reference, acceleration]))
    curve_time = np.linspace(0, DURATION, 20*(len(time)-1)+1)
    curve = np.column_stack([smooth(curve_time), smooth.derivative()(curve_time)])
    forcing = reference[:-1] @ A.T - reference[1:]
    # For this compatible reference, the scalar least-squares inverse is exact.
    feedforward = -(forcing @ B) / (B @ B)
    return Model(time, A, B, reference, forcing, feedforward, curve_time, curve)


def defects(A, B, states, actions):
    """Signed equality residual, next state minus the model prediction."""
    return states[1:] - states[:-1] @ A.T - actions[:, None] * B


def cost(reference, states, actions):
    errors = states[1:] - reference[1:]
    return 0.5 * (np.sum((errors @ Q) * errors) + R * (actions @ actions))


def qp_matrices(model, steps):
    """Use stage order z = (p1, v1, a0, p2, v2, a1, ...)."""
    A, B = model.A, model.B
    free = rollout(A, B, model.reference[0], np.zeros(steps))[1:].ravel()
    G = np.zeros((steps, 2, steps))
    D = sparse.lil_matrix((2 * steps, 3 * steps))
    rhs = np.zeros(2 * steps)
    rhs[:2] = A @ model.reference[0]
    for k in range(steps):
        rows = slice(2*k, 2*k+2)
        col = 3*k
        if k:
            G[k] = A @ G[k-1]
            D[rows, col-3:col-1] = -A
        G[k, :, k] = B
        D[rows, col:col+2] = np.eye(2)
        D[rows, col+2] = -B[:, None]
    G = G.reshape(2*steps, steps)
    weights = np.tile(Q.diagonal(), steps)
    H = G.T @ (weights[:, None] * G) + R * np.eye(steps)
    W = sparse.diags(np.tile([*Q.diagonal(), R], steps), format="csr")
    return free, G, H, W, D.tocsr(), rhs


def solve_tracking_knots(model, limit, *, running_cost=True, terminal=True):
    """Solve for (position, velocity, acceleration) at each stage of the slide 6 problem."""
    n = len(model.time) - 1
    # Row k stores (s[k+1], a[k]); D and rhs encode dynamics from the given s[0].
    _, _, _, _, D, rhs = qp_matrices(model, n)
    dynamics = LinearConstraint(D.toarray(), rhs, rhs)
    upper = np.tile([np.inf, np.inf, limit], (n, 1))
    lower = -upper
    if terminal:
        lower[-1, :2] = upper[-1, :2] = model.reference[-1]

    def objective(z):
        error = z.reshape(n, 3)[:, :2] - model.reference[1:]
        gradient = np.zeros((n, 3))
        if not running_cost:
            return 0.0, gradient.ravel()
        gradient[:, :2] = 2 * error
        return np.sum(error**2), gradient.ravel()

    result = minimize(objective, np.zeros(3*n), jac=True, method="SLSQP",
                      bounds=Bounds(lower.ravel(), upper.ravel()), constraints=dynamics,
                      options={"ftol": 1e-12, "maxiter": 250})
    if not result.success:
        raise RuntimeError(result.message)
    stages = result.x.reshape(n, 3)
    return np.vstack([model.reference[0], stages[:, :2]]), stages[:, 2]


@dataclass
class TrackingPlan:
    clipped: np.ndarray
    open_loop: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    dense_actions: np.ndarray
    limit: float


def optimize(model, limit):
    """Solve the joint state/input QP and independently solve its condensation."""
    n = len(model.feedforward)
    free, G, _, W, D, rhs = qp_matrices(model, n)
    clipped = np.clip(model.feedforward, -limit, limit)
    open_loop = rollout(model.A, model.B, model.reference[0], clipped)
    z0 = np.column_stack([open_loop[1:], clipped]).ravel()
    desired = np.column_stack([model.reference[1:], np.zeros(n)]).ravel()
    weights = W.diagonal()
    bound = np.tile([np.inf, np.inf, limit], n)
    # SLSQP accepts a dense Jacobian. D and W above retain the sparse formulation.
    joint = minimize(
        lambda z: 0.5 * ((z-desired)**2 @ weights), z0,
        jac=lambda z: weights * (z-desired), method="SLSQP",
        bounds=Bounds(-bound, bound), constraints=[LinearConstraint(D.toarray(), rhs, rhs)],
        options={"ftol": 1e-11, "maxiter": 250},
    )
    if not joint.success:
        raise RuntimeError(joint.message)
    stages = joint.x.reshape(n, 3)
    states = np.vstack([model.reference[0], stages[:, :2]])
    actions = stages[:, 2]

    sqrt_q = np.tile(np.sqrt(Q.diagonal()), n)
    design = np.vstack([sqrt_q[:, None]*G, np.sqrt(R)*np.eye(n)])
    target = np.r_[sqrt_q*(model.reference[1:].ravel()-free), np.zeros(n)]
    dense = lsq_linear(design, target, bounds=(-limit, limit), method="bvls", tol=1e-12)
    if not dense.success:
        raise RuntimeError(dense.message)
    return TrackingPlan(clipped, open_loop, actions, states, dense.x, limit)
