"""Conservative SLIP, using Beyond Basins of Attraction (Heim & Spröwitz).

https://arxiv.org/abs/1806.08081, sections II–III and appendix, Fig. 11.
Parameters: mass [kg], stiffness [N/m], resting_length [m], gravity [m/s²],
total_energy [J]. Cartesian states are [x, y, vx, vy]; the reduced state is
s = m*g*y/E at flight apex. Positive alpha puts the foot ahead of the body.
Each cycle is ballistic flight, a compressive spring stance, and flight to
the next apex. Liftoff is outward crossing of the spring's resting length,
consistent with the author's vibly/models/slip.py implementation.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix


PAPER_PARAMS = {"mass": 80.0, "stiffness": 8200.0, "resting_length": 1.0,
                "gravity": 9.81, "total_energy": 1860.0}
RETURN, INFEASIBLE, FALL, REVERSAL, NO_APEX, TIMEOUT = range(6)
STATUS_LABELS = ("Next apex", "Foot starts below ground", "Body hit the ground",
                 "Forward motion reversed", "No ascending flight after stance", "Stance time limit")


def validate_params(params):
    for name in PAPER_PARAMS:
        if not np.isfinite(params[name]) or params[name] <= 0:
            raise ValueError(f"SLIP {name} must be finite and positive.")


def apex_state(state, params, *, x=0.0):
    """Lift s in [0,1] to the positive-speed, fixed-energy apex section."""
    if not np.isfinite(state) or not 0 <= state <= 1:
        raise ValueError("Normalized apex height must lie in [0, 1].")
    e = params["total_energy"] / params["mass"]
    return np.array([x, state * e / params["gravity"], np.sqrt(2 * e * (1 - state)), 0.0])


def apex_coordinate(state, params):
    return params["mass"] * params["gravity"] * np.asarray(state)[..., 1] / params["total_energy"]


def stance_dynamics(time, state, params):
    """Cartesian acceleration relative to the fixed touchdown foot."""
    del time
    x, y, vx, vy = state
    length = max(np.hypot(x, y), 1e-12)
    force = params["stiffness"] / params["mass"] * (params["resting_length"] - length)
    return np.array([vx, vy, force * x / length, force * y / length - params["gravity"]])


def step(apex, alpha, params, *, record=False, max_step=0.02, rtol=1e-9, atol=1e-11):
    """Simulate one physical cycle. Missing next apices have next_state=NaN.

    Trace columns: time, x, y, vx, vy, foot_x, foot_y, stance flag, alpha.
    Flight is analytic. Stance uses adaptive DOP853 with directed events.
    Parameters and input arrays are never mutated.
    """
    validate_params(params)
    apex = np.asarray(apex, dtype=float)
    if apex.shape != (4,) or not np.all(np.isfinite(apex)) or abs(apex[3]) > 1e-8:
        raise ValueError("Start SLIP at a finite flight apex [x, y, vx, 0].")
    if not np.isfinite(alpha) or not -np.pi <= alpha <= np.pi:
        raise ValueError("SLIP angle of attack must lie in [-π, π].")
    g, length = params["gravity"], params["resting_length"]
    segments = []

    def flight(start, duration, elapsed, angle):
        times = np.linspace(0, duration, max(2, int(np.ceil(duration / 0.02)) + 1)) if record else np.array([duration])
        states = np.broadcast_to(start, (len(times), 4)).copy()
        states[:, 0] += start[2] * times
        states[:, 1] += start[3] * times - 0.5 * g * times**2
        states[:, 3] -= g * times
        if record:
            foot = states[:, :2] + length * np.array([np.sin(angle), -np.cos(angle)])
            segments.append(np.column_stack([times + elapsed, states, foot, np.zeros(len(times)), np.full(len(times), alpha)]))
        return states[-1]

    def result(status, end, elapsed):
        if record and not segments:
            flight(apex, 0.0, 0.0, alpha)
        return {"status": status, "state": end.copy(), "elapsed": float(elapsed),
                "next_state": float(apex_coordinate(end, params)) if status == RETURN else np.nan,
                "trace": np.concatenate(segments) if record else None}

    if apex[1] <= 0:
        return result(FALL, apex, 0)
    if apex[2] < 0:
        return result(REVERSAL, apex, 0)
    touchdown_height = length * np.cos(alpha)
    if apex[1] < touchdown_height - 1e-12:
        return result(INFEASIBLE, apex, 0)
    if touchdown_height <= 1e-12:
        duration = np.sqrt(2 * apex[1] / g)
        end = flight(apex, duration, 0, alpha)
        end[1] = 0.0
        return result(FALL, end, duration)

    touchdown_time = np.sqrt(max(0, 2 * (apex[1] - touchdown_height) / g))
    touchdown = flight(apex, touchdown_time, 0, alpha)
    foot_x = touchdown[0] + length * np.sin(alpha)
    local = touchdown.copy()
    local[0] -= foot_x
    local[1] = touchdown_height
    # A grazing/outward contact cannot load a compressive spring or create
    # an ascending flight; do not count an immediate liftoff root as a hop.
    if local[:2] @ local[2:] >= 0:
        duration = (local[3] + np.sqrt(local[3]**2 + 2 * g * local[1])) / g
        end = flight(touchdown, duration, touchdown_time, alpha)
        end[1] = 0
        return result(NO_APEX, end, touchdown_time + duration)

    def liftoff(t, state):
        return np.hypot(state[0], state[1]) - length

    def fall(t, state):
        return state[1]

    def reversal(t, state):
        # Permit exactly vertical hopping; stop once reversal exceeds roundoff.
        return state[2] + 1e-9

    liftoff.terminal, liftoff.direction = True, 1
    fall.terminal, fall.direction = True, -1
    reversal.terminal, reversal.direction = True, -1
    solution = solve_ivp(lambda t, state: stance_dynamics(t, state, params),
                         (0, 5.0), local, method="DOP853", events=(liftoff, fall, reversal),
                         max_step=max_step, rtol=rtol, atol=atol)
    if not solution.success:
        raise ValueError(f"SLIP integration failed: {solution.message}")
    world = solution.y.T.copy()
    world[:, 0] += foot_x
    elapsed = touchdown_time + solution.t[-1]
    if record:
        segments.append(np.column_stack([solution.t + touchdown_time, world,
                                         np.full(len(world), foot_x), np.zeros(len(world)),
                                         np.ones(len(world)), np.full(len(world), alpha)]))
    end = world[-1]
    if solution.t_events[1].size:
        end[1] = 0
        return result(FALL, end, elapsed)
    if solution.t_events[2].size:
        return result(REVERSAL, end, elapsed)
    if not solution.t_events[0].size:
        return result(TIMEOUT, end, elapsed)
    # Keep the leg's liftoff orientation in flight until the next apex decision.
    flight_angle = np.arctan2(foot_x - end[0], end[1])
    if end[3] <= 0:
        duration = (end[3] + np.sqrt(end[3]**2 + 2 * g * end[1])) / g
        end = flight(end, duration, elapsed, flight_angle)
        end[1] = 0
        return result(NO_APEX, end, elapsed + duration)
    duration = end[3] / g
    end = flight(end, duration, elapsed, flight_angle)
    end[3] = 0.0
    return result(RETURN, end, elapsed + duration)


@lru_cache(maxsize=4)
def _grid(state_nodes, action_nodes, parameter_items):
    params = dict(parameter_items)
    states, actions = np.asarray(state_nodes), np.asarray(action_nodes)
    next_states = np.full((len(states), len(actions)), np.nan)
    status = np.empty(next_states.shape, dtype=np.int8)
    for i, state in enumerate(states):
        apex = apex_state(state, params)
        for j, alpha in enumerate(actions):
            transition = step(apex, alpha, params)
            next_states[i, j], status[i, j] = transition["next_state"], transition["status"]
    next_states.setflags(write=False)
    status.setflags(write=False)
    return next_states, status


def build_grid(states, actions, params):
    """Build a linear-interpolation transition matrix, caching only dynamics."""
    validate_params(params)
    states, actions = np.asarray(states), np.asarray(actions)
    if len(states) < 2 or states[0] != 0 or states[-1] != 1 or np.any(np.diff(states) <= 0):
        raise ValueError("The SLIP state grid must increase from 0 to 1.")
    if len(actions) < 2 or np.any(np.diff(actions) <= 0) or actions[0] < -np.pi or actions[-1] > np.pi:
        raise ValueError("Use increasing SLIP actions within [-π, π].")
    next_states, status = _grid(tuple(states), tuple(actions), tuple((k, params[k]) for k in PAPER_PARAMS))
    count, terminal = next_states.size, len(states)
    rows = np.arange(count)
    low = np.full(count, terminal, dtype=int)
    high = low.copy()
    weights = np.zeros(count)
    active = np.flatnonzero(status.ravel() == RETURN)
    values = next_states.ravel()[active]
    if np.any((values < -1e-7) | (values > 1 + 1e-7)):
        raise ValueError("SLIP returned outside its fixed-energy section; check integration accuracy.")
    values = np.clip(values, 0, 1)  # Only roundoff at the energy boundary.
    upper = np.clip(np.searchsorted(states, values), 1, terminal - 1)
    lower = upper - 1
    low[active], high[active] = lower, upper
    weights[active] = (values - states[lower]) / (states[upper] - states[lower])
    matrix = csr_matrix((np.r_[1 - weights, weights], (np.tile(rows, 2), np.r_[low, high])),
                        shape=(count, terminal + 1))
    matrix.eliminate_zeros()
    return matrix, next_states, status


def rollout(initial_state, choose_action, params, *, steps=15):
    """Roll out continuous physics; do not snap apex states onto grid nodes."""
    current = apex_state(initial_state, params)
    states, actions, traces = [float(initial_state)], [], []
    elapsed = 0.0
    outcome = "Decision limit reached"
    for _ in range(steps):
        action = float(choose_action(states[-1]))
        transition = step(current, action, params, record=True)
        trace = transition["trace"].copy()
        trace[:, 0] += elapsed
        traces.append(trace)
        actions.append(action)
        elapsed += transition["elapsed"]
        if transition["status"] != RETURN:
            outcome = STATUS_LABELS[transition["status"]]
            break
        current = transition["state"]
        states.append(float(np.clip(transition["next_state"], 0, 1)))
    return {"states": np.array(states)[:, None], "actions": np.array(actions),
            "trace": np.concatenate(traces), "outcome": outcome}
