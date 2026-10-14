"""Assignment 2's bounded ankle controller and event-driven walker simulation.

Adapted from ../../CodeAssignments/walking_control.py, integrators.py, and the
completed instructor/inverted_pendulum_walker.py (the public model is a stub).
Kept local so the notebook does not depend on the student's completed code.
The ankle stays off until capture, then the saturated feedback controller is
latched on with the swing leg clear, as in the assignment report.
"""

from functools import lru_cache
from pathlib import Path
import json

import numpy as np

RETURN, CAPTURE, FAILURE, TIMEOUT = range(4)
RUNNING = -1


def controller_params(slope=0.06):
    """Assignment defaults, including gains and asymmetric ankle-torque bounds."""
    metadata = Path(__file__).with_name("data") / "walker_capture_region.json"
    params = json.loads(metadata.read_text())["params"]
    return params | {"incline": float(slope)}


@lru_cache(maxsize=4)
def capture_region(slope=0.06):
    """Reuse the assignment's validated default RoA; recompute changed slopes.

    The cached artifact contains the original classification as well as the
    eroded capture boundary. All other controller/physical parameters are
    fixed by controller_params. Returned arrays must be treated as read-only.
    """
    if float(slope) == 0.06:
        path = Path(__file__).with_name("data") / "walker_capture_region.npz"
        with np.load(path) as saved:
            return {key: saved[key] for key in saved.files}
    return estimate_standing_region(controller_params(slope), angle_count=481, velocity_count=801)

def generate_params():
    """Return physical parameters; control bounds belong in the experiment."""
    return {
        "gravity": 9.81,
        "length": 1.0,
        "mass": 1.0,
        "incline": 0.06,
        "angle_of_attack": np.pi / 8,
        "ankle_torque": 0.0,
    }


def dynamics(time, state, params):
    """Return [angular velocity, angular acceleration], allowing batch states."""
    del time
    mass = params["mass"]
    length = params["length"]
    angular_acceleration = (
        mass * params["gravity"] * length * np.sin(state[0])
        + params.get("ankle_torque", 0.0)
    ) / (mass * length**2)
    return np.array([state[1], angular_acceleration])


def contact_angles(params):
    """Return the post-impact angle and forward touchdown angle."""
    return (
        params["incline"] - params["angle_of_attack"],
        params["incline"] + params["angle_of_attack"],
    )


def standing_angle_bounds(params):
    """Unwrapped stance angles for which the hub is above the inclined ground."""
    return params["incline"] - np.pi / 2, params["incline"] + np.pi / 2


def calculate_ground_clearance(state, params):
    """Signed perpendicular distance of the point-mass hub from the ground."""
    return params["length"] * np.cos(state[0] - params["incline"])


def ground_guard_value(state, params):
    """Nonnegative at/beyond first ground contact, with no angle wrapping.

    Checking clearance alone would incorrectly admit an extra full revolution.
    The allowed connected angle interval is the one containing upright stance.
    """
    return np.abs(state[0] - params["incline"]) - np.pi / 2


def event_guard(previous_state, next_state, params):
    """Detect forward swing-foot touchdown, not backward rocking contacts."""
    _, touchdown_angle = contact_angles(params)
    return (
        (previous_state[0] < touchdown_angle)
        & (next_state[0] >= touchdown_angle)
        & (next_state[1] > 0)
    )


def event_dynamics(state, params):
    """Apply a forward plastic collision about the new foot.

    Use the angle of the completed step, before selecting a subsequent action.
    A negative outgoing velocity is retained, not silently clipped to zero.
    """
    angle_of_attack = params["angle_of_attack"]
    return np.array(
        [state[0] - 2 * angle_of_attack, np.cos(2 * angle_of_attack) * state[1]]
    )


def calculate_energy(state, params):
    """Return kinetic and stance-relative potential energy."""
    kinetic = 0.5 * params["mass"] * (params["length"] * state[1]) ** 2
    potential = params["mass"] * params["gravity"] * params["length"] * np.cos(state[0])
    return kinetic, potential


def rk4(dynamics, time, state, timestep, params):
    """Advance one fourth-order Runge-Kutta step."""
    k1 = dynamics(time, state, params)
    k2 = dynamics(time + timestep / 2, state + timestep * k1 / 2, params)
    k3 = dynamics(time + timestep / 2, state + timestep * k2 / 2, params)
    k4 = dynamics(time + timestep, state + timestep * k3, params)
    return state + timestep * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def capture_margin(region, states):
    """Signed distance inside the interpolated empirical standing region."""
    angle, velocity = states
    lower = np.interp(angle, region["angles"], region["lower"])
    upper = np.interp(angle, region["angles"], region["upper"])
    margin = np.minimum(velocity - lower, upper - velocity)
    left, right = region["ground_angles"]
    valid = (
        (angle >= region["angles"][0])
        & (angle <= region["angles"][-1])
        & (angle > left)
        & (angle < right)
        & np.isfinite(margin)
    )
    return np.where(valid, margin, -1.0)


def compute_ankle_torque(states, params):
    """Cancel gravity, add stable linear feedback, then enforce torque bounds."""
    frequency = np.sqrt(params["gravity"] / params["length"])
    normalized_torque = (
        -np.sin(states[0])
        - params["position_gain"] * states[0]
        - params["velocity_gain"] * states[1] / frequency
    )
    torque_scale = params["mass"] * params["gravity"] * params["length"]
    return np.clip(
        torque_scale * normalized_torque, params["torque_min"], params["torque_max"]
    )


def has_settled(states, params):
    """Check a small invariant ellipse of the unsaturated local closed loop."""
    frequency = np.sqrt(params["gravity"] / params["length"])
    radius = np.hypot(
        np.sqrt(params["position_gain"]) * states[0], states[1] / frequency
    )
    return radius <= params["settle_tolerance"]


def simulate_balance(initial_states, params, record=False):
    """Simulate the bounded controller, latched on with the swing foot clear."""
    states = np.asarray(initial_states, dtype=float).reshape(2, -1).copy()
    if record and states.shape[1] != 1:
        raise ValueError("Only a single balance trajectory can be recorded.")
    status = np.full(states.shape[1], TIMEOUT, dtype=np.int8)
    active = np.ones(states.shape[1], dtype=bool)
    trace = [] if record else None
    for index in range(
        int(np.ceil(params["balance_horizon"] / params["timestep"])) + 1
    ):
        elapsed = index * params["timestep"]
        if record:
            trace.append(
                (
                    elapsed,
                    *states[:, 0],
                    float(compute_ankle_torque(states[:, 0], params)),
                    params["angle_of_attack"],
                    0,
                )
            )
        settled = has_settled(states, params) & active
        failed = (ground_guard_value(states, params) >= 0) & active
        status[settled] = CAPTURE
        status[failed] = FAILURE
        active[settled | failed] = False
        if not np.any(active) or elapsed >= params["balance_horizon"]:
            break
        selected = states[:, active]
        # Sample and hold once per integration timestep, as the assignment asks.
        controlled_params = params | {
            "ankle_torque": compute_ankle_torque(selected, params)
        }
        next_states = rk4(
            dynamics, elapsed, selected, params["timestep"], controlled_params
        )
        ground_contact = ground_guard_value(next_states, params) >= 0
        active_indices = np.flatnonzero(active)
        if np.any(ground_contact):
            contact_params = controlled_params | {
                "ankle_torque": controlled_params["ankle_torque"][ground_contact]
            }
            contact_times, contact_states = locate_event(
                selected[:, ground_contact],
                np.full(np.count_nonzero(ground_contact), params["timestep"]),
                contact_params,
                lambda state: ground_guard_value(state, params),
                params["event_iterations"],
            )
            next_states[:, ground_contact] = contact_states
            failed_indices = active_indices[ground_contact]
            status[failed_indices] = FAILURE
            active[failed_indices] = False
            if record:
                trace.append(
                    (
                        elapsed + contact_times[0],
                        *contact_states[:, 0],
                        float(contact_params["ankle_torque"][0]),
                        params["angle_of_attack"],
                        0,
                    )
                )
        states[:, active_indices] = next_states
        if record and np.any(ground_contact):
            break
    return status, states, trace


def estimate_standing_region(params, angle_count=241, velocity_count=401):
    """Grid initial states, simulate the actual bounded controller, retain an interior."""
    # Wider than the walking-policy speed sweep, so the RoA does not disappear
    # through the edge of the figure. Sample invalid angles to show the floor.
    speed_limit = 2 * np.sqrt(params["gravity"] / params["length"])
    ground_angles = standing_angle_bounds(params)
    angles = np.linspace(ground_angles[0] - 0.15, ground_angles[1] + 0.15, angle_count)
    velocities = np.linspace(-speed_limit, speed_limit, velocity_count)
    angle_grid, velocity_grid = np.meshgrid(angles, velocities, indexing="ij")
    status, _, _ = simulate_balance(
        np.array([angle_grid.ravel(), velocity_grid.ravel()]), params
    )
    classification = status.reshape(angle_count, velocity_count)
    lower = np.full(angle_count, np.nan)
    upper = np.full(angle_count, np.nan)
    # One extra half-cell on each side avoids claiming the first failed cell.
    padding = 0.5 * (velocities[1] - velocities[0])
    for index, row in enumerate(classification):
        successful = np.flatnonzero(row == CAPTURE)
        if successful.size >= 3:
            if np.any(np.diff(successful) != 1):
                raise RuntimeError(
                    "The RoA has disconnected velocity intervals; revise its representation."
                )
            lower[index] = velocities[successful[0]] + padding
            upper[index] = velocities[successful[-1]] - padding
    return {
        "angles": angles,
        "velocities": velocities,
        "status": classification,
        "lower": lower,
        "upper": upper,
        "ground_angles": ground_angles,
    }


def locate_event(states, durations, params, guard, iterations):
    """Bisect a bracket whose guard changes from negative to nonnegative."""
    lower = np.zeros_like(durations)
    upper = np.asarray(durations).copy()
    for _ in range(iterations):
        middle = (lower + upper) / 2
        midpoint = rk4(dynamics, 0.0, states, middle, params)
        crossed = guard(midpoint) >= 0
        upper = np.where(crossed, middle, upper)
        lower = np.where(crossed, lower, middle)
    durations = (lower + upper) / 2
    return durations, rk4(dynamics, 0.0, states, durations, params)


def simulate_transitions(velocities, actions, region, params, record=False):
    """Numerically integrate section -> impact -> section, stopping early at capture.

    Each batch member has its own clock. At a located impact its next integration
    step begins at the impact, so no unintegrated time is silently discarded.
    A backward-moving post-impact trajectory is outside this forward-walking map.
    """
    velocities, actions = np.broadcast_arrays(velocities, actions)
    shape = velocities.shape
    velocities = velocities.ravel()
    actions = actions.ravel()
    count = velocities.size
    if record and count != 1:
        raise ValueError("Only a single transition can be recorded.")
    states = np.array([np.zeros(count), velocities.copy()])
    elapsed = np.zeros(count)
    impacts = np.zeros(count, dtype=np.int8)
    status = np.full(count, RUNNING, dtype=np.int8)
    passive_params = params | {"ankle_torque": 0.0}
    trace = [(0.0, *states[:, 0], 0.0, actions[0], 1)] if record else None

    while np.any(status == RUNNING):
        active = status == RUNNING
        captured = active & (capture_margin(region, states) >= 0)
        status[captured] = CAPTURE
        failed = (
            active
            & ~captured
            & ((ground_guard_value(states, params) >= 0) | (states[1] <= 0))
        )
        status[failed] = FAILURE
        status[(status == RUNNING) & (elapsed >= params["transition_horizon"])] = (
            TIMEOUT
        )
        selected = np.flatnonzero(status == RUNNING)
        if selected.size == 0:
            break
        start = states[:, selected]
        durations = np.minimum(
            params["timestep"], params["transition_horizon"] - elapsed[selected]
        )
        endpoint = rk4(dynamics, 0.0, start, durations, passive_params)
        event = np.full(selected.size, RUNNING, dtype=np.int8)
        touchdown_angle = params["incline"] + actions[selected]

        # Hard events occur in separate stance phases, except the fall guard.
        candidates = [
            ((impacts[selected] == 0) & (endpoint[0] >= touchdown_angle), 4, 0),
            ((impacts[selected] == 1) & (start[0] < 0) & (endpoint[0] >= 0), RETURN, 1),
            ((start[1] > 0) & (endpoint[1] <= 0), FAILURE, 2),
            (ground_guard_value(endpoint, params) >= 0, FAILURE, 3),
        ]
        for mask, event_code, guard_kind in candidates:
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                continue
            if guard_kind == 0:
                threshold = touchdown_angle[indices]
                guard = lambda state, threshold=threshold: state[0] - threshold
            elif guard_kind == 1:
                guard = lambda state: state[0]
            elif guard_kind == 2:
                guard = lambda state: -state[1]
            else:
                guard = lambda state: ground_guard_value(state, params)
            # Another earlier guard may already have shortened this segment.
            bracketed = guard(endpoint[:, indices]) >= 0
            indices = indices[bracketed]
            if indices.size == 0:
                continue
            if guard_kind == 0:
                threshold = touchdown_angle[indices]
                guard = lambda state, threshold=threshold: state[0] - threshold
            event_time, event_state = locate_event(
                start[:, indices],
                durations[indices],
                passive_params,
                guard,
                params["event_iterations"],
            )
            durations[indices] = event_time
            endpoint[:, indices] = event_state
            event[indices] = event_code

        capture_indices = np.flatnonzero(capture_margin(region, endpoint) >= 0)
        if capture_indices.size:
            capture_time, capture_state = locate_event(
                start[:, capture_indices],
                durations[capture_indices],
                passive_params,
                lambda state: capture_margin(region, state),
                params["event_iterations"],
            )
            durations[capture_indices] = capture_time
            endpoint[:, capture_indices] = capture_state
            event[capture_indices] = CAPTURE

        elapsed[selected] += durations
        states[:, selected] = endpoint
        if record:
            trace.append((elapsed[0], *states[:, 0], 0.0, actions[0], 1))
        touchdown = event == 4
        if np.any(touchdown):
            impact_indices = selected[touchdown]
            states[:, impact_indices] = event_dynamics(
                states[:, impact_indices],
                params | {"angle_of_attack": actions[impact_indices]},
            )
            impacts[impact_indices] += 1
            if record:
                trace.append((elapsed[0], *states[:, 0], 0.0, actions[0], 1))
        terminal = (event != RUNNING) & ~touchdown
        status[selected[terminal]] = event[terminal]

    return {
        "states": states.reshape((2,) + shape),
        "status": status.reshape(shape),
        "impacts": impacts.reshape(shape),
        "elapsed": elapsed.reshape(shape),
        "trace": trace,
    }

